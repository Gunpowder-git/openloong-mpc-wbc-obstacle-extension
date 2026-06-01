/*
Recovery diagnostic: stand the robot, apply a push at t=5s, then keep the
stand controller running and observe whether it self-rights. Optionally ramps
a base-height "get-up" offset (the approach used by manual_control_mode) to
test whether commanding base height can lift a fallen robot.

Usage: recovery_test <pushForceN> <getup:0|1>
*/
#include <mujoco/mujoco.h>
#include "MJ_interface.h"
#include "PVT_ctrl.h"
#include "data_bus.h"
#include "pino_kin_dyn.h"
#include "useful_math.h"
#include "wbc_priority.h"
#include "mpc.h"
#include "gait_scheduler.h"
#include "foot_placement.h"
#include "joystick_interpreter.h"
#include "StateEst.h"
#include <iostream>
#include <cstdio>
#include <cmath>

const double dt = 0.001;
const double dt_200Hz = 0.005;

int main(int argc, char **argv)
{
    double pushForce = (argc > 1) ? atof(argv[1]) : 400.0;
    bool useGetup    = (argc > 2) ? (atoi(argv[2]) != 0) : false;

    char error[1000] = "load fail";
    mjModel *mj_model = mj_loadXML("../models/scene.xml", 0, error, 1000);
    if (!mj_model) { fprintf(stderr, "%s\n", error); return 1; }
    mjData *mj_data = mj_makeData(mj_model);

    MJ_Interface mj_interface(mj_model, mj_data);
    Pin_KinDyn kinDynSolver("../models/AzureLoong.urdf");
    DataBus RobotState(kinDynSolver.model_nv);
    WBC_priority WBC_solv(kinDynSolver.model_nv, 18, 22, 0.7, mj_model->opt.timestep);
    MPC MPC_solv(dt_200Hz);
    GaitScheduler gaitScheduler(0.4, mj_model->opt.timestep);
    PVT_Ctr pvtCtr(mj_model->opt.timestep, "../common/joint_ctrl_config.json");
    FootPlacement footPlacement;
    JoyStickInterpreter jsInterp(mj_model->opt.timestep);
    StateEst StateModule(0.001);

    double stand_legLength = 1.01, foot_height = 0.07;
    const int robot_nv = kinDynSolver.model_nv;
    RobotState.width_hips = 0.229;
    footPlacement.kp_vx = 0.03; footPlacement.kp_vy = 0.03; footPlacement.kp_wz = 0.03;
    footPlacement.stepHeight = 0.12; footPlacement.legLength = stand_legLength;

    std::vector<double> motors_vel_des(robot_nv - 6, 0), motors_tau_des(robot_nv - 6, 0);

    Eigen::Vector3d fe_l_pos_L_des = {-0.018, 0.113, -stand_legLength};
    Eigen::Vector3d fe_r_pos_L_des = {-0.018, -0.116, -stand_legLength};
    Eigen::Matrix3d fe_l_rot_des = eul2Rot(0,-0.008,0), fe_r_rot_des = eul2Rot(0,-0.008,0);
    Eigen::VectorXd hd_l_des(7), hd_r_des(7);
    hd_l_des << 0.475,-1.12,1.9,0.86,-0.356,0,0;
    hd_r_des << -0.475,-1.12,-1.9,0.86,0.356,0,0;
    auto resLeg = kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des, fe_r_rot_des, fe_r_pos_L_des);
    Eigen::VectorXd qIniDes = Eigen::VectorXd::Zero(mj_model->nq, 1);
    qIniDes.block(7,0,mj_model->nq-7,1) = resLeg.jointPosRes;
    qIniDes.block(7,0,7,1) = hd_l_des; qIniDes.block(14,0,7,1) = hd_r_des;
    WBC_solv.setQini(qIniDes, RobotState.q);

    int baseBody = mj_name2id(mj_model, mjOBJ_BODY, "base_link");
    int MPC_count = 0;
    double openLoopCtrTime = 3, simTime = 0, nextLog = 0;
    bool pushed = false;
    double getupStart = -1;

    printf("# pushForce=%.0f useGetup=%d baseBody=%d\n", pushForce, useGetup, baseBody);
    printf("t,x,z,roll,pitch\n");

    while (simTime < 20.0)
    {
        // apply push between t=5.0 and t=5.3 (lateral + slight down to topple)
        for (int b=0;b<6;b++) mj_data->xfrc_applied[baseBody*6+b]=0;
        if (simTime > 5.0 && simTime < 5.3) {
            mj_data->xfrc_applied[baseBody*6+1] = pushForce;   // +Y push
            mj_data->xfrc_applied[baseBody*6+0] = pushForce*0.4;
        }

        mj_step(mj_model, mj_data);
        simTime = mj_data->time;
        mj_interface.updateSensorValues();
        mj_interface.dataBusWrite(RobotState);
        if (simTime > 1 && StateModule.flag_init) StateModule.init(RobotState);

        StateModule.set(RobotState); StateModule.update(); StateModule.get(RobotState);
        kinDynSolver.dataBusRead(RobotState); kinDynSolver.computeJ_dJ();
        kinDynSolver.computeDyn(); kinDynSolver.dataBusWrite(RobotState);
        StateModule.setF(RobotState); StateModule.updateF(); StateModule.getF(RobotState);

        if (simTime >= openLoopCtrTime && simTime < openLoopCtrTime + 0.002)
            RobotState.motionState = DataBus::Stand;

        // detect fall and start getup ramp
        if (useGetup && getupStart < 0 && simTime > 6.0 &&
            (std::fabs(RobotState.base_rpy(0)) > 1.0 || RobotState.base_pos(2) < 0.55))
            getupStart = simTime;

        jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));

        MPC_solv.disable();

        if (simTime <= openLoopCtrTime)
        {
            WBC_solv.setQini(qIniDes, RobotState.q);
            WBC_solv.fe_l_pos_des_W = RobotState.fe_l_pos_W;
            WBC_solv.fe_r_pos_des_W = RobotState.fe_r_pos_W;
            WBC_solv.fe_l_rot_des_W = RobotState.fe_l_rot_W;
            WBC_solv.fe_r_rot_des_W = RobotState.fe_r_rot_W;
            WBC_solv.pCoMDes = RobotState.pCoM_W;
        }

        MPC_count++;
        if (MPC_count > (dt_200Hz/dt - 1)) { MPC_solv.dataBusRead(RobotState); MPC_solv.cal(); MPC_count=0; }

        RobotState.Fr_ff = Eigen::VectorXd::Zero(12);
        RobotState.des_ddq = Eigen::VectorXd::Zero(RobotState.model_nv);
        RobotState.des_dq = Eigen::VectorXd::Zero(RobotState.model_nv);
        RobotState.des_delta_q = Eigen::VectorXd::Zero(RobotState.model_nv);
        RobotState.base_rpy_des << 0.0, 0.0, jsInterp.thetaZ;
        RobotState.base_pos_des = RobotState.js_pos_des;
        double zCmd = stand_legLength + foot_height;
        if (useGetup && getupStart > 0) {
            double at = simTime - getupStart;
            // mimic manual-mode getup: dip then rise base height target
            if (at < 2.0) zCmd += -0.20;
            else if (at < 5.0) zCmd += -0.20 + 0.20*((at-2.0)/3.0);
        }
        RobotState.base_pos_des(2) = zCmd;
        RobotState.Fr_ff << 0,0,370,0,0,0, 0,0,370,0,0,0;

        WBC_solv.dataBusRead(RobotState);
        WBC_solv.computeDdq(kinDynSolver);
        WBC_solv.computeTau();
        WBC_solv.dataBusWrite(RobotState);

        if (simTime <= openLoopCtrTime)
        {
            Eigen::VectorXd temp = resLeg.jointPosRes;
            temp.block(0,0,7,1)=hd_l_des; temp.block(7,0,7,1)=hd_r_des;
            RobotState.motors_pos_des = eigen2std(temp);
            RobotState.motors_vel_des = motors_vel_des;
            RobotState.motors_tor_des = motors_tau_des;
        }
        else
        {
            Eigen::Matrix<double,1,nx> L_diag; Eigen::Matrix<double,1,nu> K_diag;
            L_diag << 1.0,1.0,1.0, 1e-3,100.0,1.0, 1e-3,1e-3,1e-3, 1,100.0,1.0;
            K_diag << 1.0,1.0,1.0, 1.0,1,1.0, 1.0,1.0,1.0, 1.0,1,1.0, 1.0;
            MPC_solv.set_weight(1e-6, L_diag, K_diag);
            Eigen::VectorXd pos_des = kinDynSolver.integrateDIY(RobotState.q, RobotState.wbc_delta_q_final);
            RobotState.motors_pos_des = eigen2std(pos_des.block(7,0,robot_nv-6,1));
            RobotState.motors_vel_des = eigen2std(RobotState.wbc_dq_final);
            RobotState.motors_tor_des = eigen2std(RobotState.wbc_tauJointRes);
        }

        pvtCtr.dataBusRead(RobotState);
        if (simTime <= openLoopCtrTime) pvtCtr.calMotorsPVT(110.0/1000.0/180.0*3.1415);
        else {
            pvtCtr.setJointPD(400,15,"J_hip_l_roll"); pvtCtr.setJointPD(200,10,"J_hip_l_yaw");
            pvtCtr.setJointPD(300,10,"J_hip_l_pitch"); pvtCtr.setJointPD(300,14,"J_knee_l_pitch");
            pvtCtr.setJointPD(300,18,"J_ankle_l_pitch"); pvtCtr.setJointPD(300,16,"J_ankle_l_roll");
            pvtCtr.setJointPD(400,15,"J_hip_r_roll"); pvtCtr.setJointPD(200,10,"J_hip_r_yaw");
            pvtCtr.setJointPD(300,10,"J_hip_r_pitch"); pvtCtr.setJointPD(300,14,"J_knee_r_pitch");
            pvtCtr.setJointPD(300,18,"J_ankle_r_pitch"); pvtCtr.setJointPD(300,16,"J_ankle_r_roll");
            pvtCtr.calMotorsPVT();
        }
        pvtCtr.dataBusWrite(RobotState);
        mj_interface.setMotorsTorque(RobotState.motors_tor_out);

        if (simTime >= nextLog) {
            printf("%.2f,%.3f,%.3f,%.3f,%.3f\n", simTime, RobotState.base_pos(0),
                   RobotState.base_pos(2), RobotState.base_rpy(0), RobotState.base_rpy(1));
            nextLog += 0.25;
        }
    }
    printf("# FINAL z=%.3f roll=%.3f pitch=%.3f  (upright if z>0.9 & |roll|<0.3)\n",
           RobotState.base_pos(2), RobotState.base_rpy(0), RobotState.base_rpy(1));
    return 0;
}
