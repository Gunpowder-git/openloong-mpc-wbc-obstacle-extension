/*
Headless verification harness for OpenLoong dyn-control.
Runs the EXACT MPC + WBC + gait + foot-placement + PVT pipeline used by the
walking demos, but with scripted velocity commands and NO GLFW window, then
prints the robot base trajectory to stdout as CSV. Used to objectively measure
standing / walking / turning / slope stability and obstacle-course progress.

Usage:
  headless_test <scene.xml> <scenario> <duration_s> [vx] [wz]
  scenario: stand | walk | turn | fwd_then_turn | walkslope
*/
#include <mujoco/mujoco.h>
#include "MJ_interface.h"
#include "PVT_ctrl.h"
#include "data_logger.h"
#include "data_bus.h"
#include "pino_kin_dyn.h"
#include "useful_math.h"
#include "wbc_priority.h"
#include "mpc.h"
#include "gait_scheduler.h"
#include "foot_placement.h"
#include "joystick_interpreter.h"
#include "StateEst.h"
#include <string>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>

const double dt = 0.001;
const double dt_200Hz = 0.005;

static double clampValue(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

int main(int argc, char **argv)
{
    std::string scenePath = (argc > 1) ? argv[1] : "../models/scene.xml";
    std::string scenario  = (argc > 2) ? argv[2] : "walk";
    double simEndTime     = (argc > 3) ? atof(argv[3]) : 30.0;
    double vxTarget       = (argc > 4) ? atof(argv[4]) : 0.4;
    double wzTarget       = (argc > 5) ? atof(argv[5]) : 0.3;

    char error[1000] = "Could not load binary model";
    mjModel *mj_model = mj_loadXML(scenePath.c_str(), 0, error, 1000);
    if (!mj_model) { fprintf(stderr, "MODEL LOAD ERROR: %s\n", error); return 1; }
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

    double stand_legLength = 1.01;
    double foot_height = 0.07;

    const int robot_nq = kinDynSolver.model_nv + 1;
    const int robot_nv = robot_nq - 1;

    RobotState.width_hips = 0.229;
    footPlacement.kp_vx = 0.03;
    footPlacement.kp_vy = 0.03;
    footPlacement.kp_wz = 0.03;
    footPlacement.stepHeight = 0.12;
    footPlacement.legLength = stand_legLength;

    std::vector<double> motors_vel_des(robot_nv - 6, 0);
    std::vector<double> motors_tau_des(robot_nv - 6, 0);

    Eigen::Vector3d fe_l_pos_L_des = {-0.018, 0.113, -stand_legLength};
    Eigen::Vector3d fe_r_pos_L_des = {-0.018, -0.116, -stand_legLength};
    Eigen::Vector3d fe_l_eul_L_des = {-0.000, -0.008, -0.000};
    Eigen::Vector3d fe_r_eul_L_des = {0.000, -0.008, 0.000};
    Eigen::Matrix3d fe_l_rot_des = eul2Rot(fe_l_eul_L_des(0), fe_l_eul_L_des(1), fe_l_eul_L_des(2));
    Eigen::Matrix3d fe_r_rot_des = eul2Rot(fe_r_eul_L_des(0), fe_r_eul_L_des(1), fe_r_eul_L_des(2));

    Eigen::VectorXd hd_l_des(7), hd_r_des(7);
    hd_l_des << 0.475, -1.12, 1.9, 0.86, -0.356, 0, 0;
    hd_r_des << -0.475, -1.12, -1.9, 0.86, 0.356, 0, 0;

    auto resLeg = kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des, fe_r_rot_des, fe_r_pos_L_des);
    Eigen::VectorXd qIniDes = Eigen::VectorXd::Zero(mj_model->nq, 1);
    qIniDes.block(7, 0, mj_model->nq - 7, 1) = resLeg.jointPosRes;
    qIniDes.block(7, 0, 7, 1) = hd_l_des;
    qIniDes.block(14, 0, 7, 1) = hd_r_des;
    WBC_solv.setQini(qIniDes, RobotState.q);

    int MPC_count = 0;
    double openLoopCtrTime = 3;
    double simTime = mj_data->time;

    bool walkStarted = false;
    bool fellWarned = false;
    double minZ = 1e9, maxRoll = 0, maxPitch = 0;

    // slope adaptation prototype
    bool slopeAdapt = getenv("SLOPE_ADAPT") != nullptr;
    double slopeGainBase = getenv("SG_BASE") ? atof(getenv("SG_BASE")) : 1.0;
    double slopeGainFoot = getenv("SG_FOOT") ? atof(getenv("SG_FOOT")) : 1.0;
    int ankleL = mj_name2id(mj_model, mjOBJ_BODY, "Link_ankle_l_roll");
    int ankleR = mj_name2id(mj_model, mjOBJ_BODY, "Link_ankle_r_roll");
    double terrainPitchF = 0.0;
    bool terrFollow = getenv("TERR_FOLLOW") != nullptr;
    double terrainZF = 0.0;
    double pushForce = getenv("PUSH") ? atof(getenv("PUSH")) : 0.0;
    bool recoverStep = getenv("RECOVER") != nullptr;
    int baseBodyId = mj_name2id(mj_model, mjOBJ_BODY, "base_link");

    printf("# scene=%s scenario=%s dur=%.1f vx=%.2f wz=%.2f\n",
           scenePath.c_str(), scenario.c_str(), simEndTime, vxTarget, wzTarget);
    printf("t,x,y,z,roll,pitch,yaw,vx_des,wz_des,feLz,feRz,ncon\n");

    double nextLog = 0.0;

    while (simTime < simEndTime)
    {
        // optional push to test disturbance recovery (t in [6.0,6.2])
        for (int b = 0; b < 6; b++) mj_data->xfrc_applied[baseBodyId*6 + b] = 0;
        if (pushForce > 0 && simTime > 6.0 && simTime < 6.2)
            mj_data->xfrc_applied[baseBodyId*6 + 1] = pushForce; // +Y lateral push

        mj_step(mj_model, mj_data);
        simTime = mj_data->time;
        mj_interface.updateSensorValues();
        mj_interface.dataBusWrite(RobotState);

        if (simTime > 1 && StateModule.flag_init)
            StateModule.init(RobotState);

        // ----- scripted commands -----
        if (simTime > openLoopCtrTime)
        {
            // start walking at t = openLoopCtrTime + 1
            if (!walkStarted && simTime > openLoopCtrTime + 1.0 && scenario != "stand")
            {
                if (RobotState.motionState == DataBus::Stand)
                {
                    gaitScheduler.start();
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                    RobotState.motionState = DataBus::Walk;
                    walkStarted = true;
                }
            }
            // apply velocity commands at t = openLoopCtrTime + 2
            if (walkStarted && simTime > openLoopCtrTime + 2.0)
            {
                if (scenario == "walk" || scenario == "walkslope")
                    jsInterp.setVxDesLPara(vxTarget, 2.0);
                else if (scenario == "turn")
                    jsInterp.setWzDesLPara(wzTarget, 1.0);   // vx stays 0 -> pure turn
                else if (scenario == "fwd_then_turn")
                {
                    if (simTime < openLoopCtrTime + 8.0)
                        jsInterp.setVxDesLPara(vxTarget, 2.0);
                    else
                        jsInterp.setWzDesLPara(wzTarget, 1.0);
                }
            }
        }

        // recovery stepping: if standing and a disturbance tilts the robot, start
        // stepping (zero commanded velocity) so foot placement can catch balance.
        if (recoverStep && simTime > openLoopCtrTime + 0.5 &&
            RobotState.motionState == DataBus::Stand &&
            (std::fabs(RobotState.base_rpy(0)) > 0.18 || std::fabs(RobotState.base_rpy(1)) > 0.18) &&
            RobotState.base_pos(2) > 0.6)
        {
            gaitScheduler.start();
            jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
            jsInterp.setVxDesLPara(0.0, 0.3);
            jsInterp.setWzDesLPara(0.0, 0.3);
            RobotState.motionState = DataBus::Walk;
            walkStarted = true;
        }

        StateModule.set(RobotState);
        StateModule.update();
        StateModule.get(RobotState);

        kinDynSolver.dataBusRead(RobotState);
        kinDynSolver.computeJ_dJ();
        kinDynSolver.computeDyn();
        kinDynSolver.dataBusWrite(RobotState);

        StateModule.setF(RobotState);
        StateModule.updateF();
        StateModule.getF(RobotState);

        if (simTime >= openLoopCtrTime && simTime < openLoopCtrTime + 0.002)
            RobotState.motionState = DataBus::Stand;

        if (RobotState.motionState == DataBus::Walk2Stand || simTime <= openLoopCtrTime)
            jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));

        if (RobotState.motionState == DataBus::Walk || RobotState.motionState == DataBus::Walk2Stand)
        {
            jsInterp.step();
            RobotState.js_pos_des(2) = stand_legLength + foot_height;
            jsInterp.dataBusWrite(RobotState);
            MPC_solv.enable();
            gaitScheduler.dataBusRead(RobotState);
            gaitScheduler.step();
            gaitScheduler.dataBusWrite(RobotState);
            footPlacement.dataBusRead(RobotState);
            footPlacement.getSwingPos();
            footPlacement.dataBusWrite(RobotState);
        }
        else
            MPC_solv.disable();

        if (simTime <= openLoopCtrTime || RobotState.motionState == DataBus::Walk2Stand)
        {
            WBC_solv.setQini(qIniDes, RobotState.q);
            WBC_solv.fe_l_pos_des_W = RobotState.fe_l_pos_W;
            WBC_solv.fe_r_pos_des_W = RobotState.fe_r_pos_W;
            WBC_solv.fe_l_rot_des_W = RobotState.fe_l_rot_W;
            WBC_solv.fe_r_rot_des_W = RobotState.fe_r_rot_W;
            WBC_solv.pCoMDes = RobotState.pCoM_W;
        }

        MPC_count++;
        if (MPC_count > (dt_200Hz / dt - 1)) {
            MPC_solv.dataBusRead(RobotState);
            MPC_solv.cal();
            MPC_count = 0;
        }

        if (RobotState.motionState == DataBus::Walk || RobotState.motionState == DataBus::Walk2Stand)
            MPC_solv.dataBusWrite(RobotState);
        else {
            RobotState.Fr_ff = Eigen::VectorXd::Zero(12);
            RobotState.des_ddq = Eigen::VectorXd::Zero(RobotState.model_nv);
            RobotState.des_dq = Eigen::VectorXd::Zero(RobotState.model_nv);
            RobotState.des_delta_q = Eigen::VectorXd::Zero(RobotState.model_nv);
            RobotState.base_rpy_des << 0.0, 0.0, jsInterp.thetaZ;
            RobotState.base_pos_des = RobotState.js_pos_des;
            RobotState.base_pos_des(2) = stand_legLength + foot_height;
            RobotState.Fr_ff << 0,0,370,0,0,0, 0,0,370,0,0,0;
        }

        // ---- terrain-height following: keep base at constant height above local ground ----
        if (terrFollow && simTime > openLoopCtrTime) {
            double zL = mj_data->xpos[3*ankleL+2], zR = mj_data->xpos[3*ankleR+2];
            double groundZ = std::min(zL, zR) - 0.07;   // sole below ankle
            if (groundZ < 0) groundZ = 0;
            terrainZF += 0.004 * (groundZ - terrainZF);
            RobotState.base_pos_des(2) = stand_legLength + foot_height + terrainZF;
        }

        // ---- slope adaptation prototype ----
        if (slopeAdapt && RobotState.motionState == DataBus::Walk) {
            double yaw = RobotState.base_rpy(2);
            double fx = std::cos(yaw), fy = std::sin(yaw);
            double *pL = mj_data->xpos + 3*ankleL;
            double *pR = mj_data->xpos + 3*ankleR;
            double sL = pL[0]*fx + pL[1]*fy, sR = pR[0]*fx + pR[1]*fy;
            double ds = sL - sR, dz = pL[2] - pR[2];
            double inst = (std::fabs(ds) > 0.06) ? std::atan2(dz, ds) : terrainPitchF;
            // heavy low-pass
            terrainPitchF += 0.01 * (inst - terrainPitchF);
            double tp = clampValue(terrainPitchF, -0.45, 0.45);
            // uphill (tp>0): lean body forward (pitch +) and tilt feet to match slope
            RobotState.base_rpy_des(1) = clampValue(RobotState.base_rpy_des(1) + slopeGainBase*tp, -0.40, 0.40);
            RobotState.swing_fe_rpy_des_W(1) = slopeGainFoot * tp;
        }

        WBC_solv.dataBusRead(RobotState);
        WBC_solv.computeDdq(kinDynSolver);
        WBC_solv.computeTau();
        WBC_solv.dataBusWrite(RobotState);

        if (simTime <= openLoopCtrTime)
        {
            Eigen::VectorXd temp = resLeg.jointPosRes;
            temp.block(0, 0, 7, 1) = hd_l_des;
            temp.block(7, 0, 7, 1) = hd_r_des;
            RobotState.motors_pos_des = eigen2std(temp);
            RobotState.motors_vel_des = motors_vel_des;
            RobotState.motors_tor_des = motors_tau_des;
        }
        else
        {
            Eigen::Matrix<double, 1, nx> L_diag;
            Eigen::Matrix<double, 1, nu> K_diag;
            L_diag << 1.0, 1.0, 1.0, 1e-3, 100.0, 1.0, 1e-3, 1e-3, 1e-3, 1, 100.0, 1.0;
            K_diag << 1.0,1.0,1.0, 1.0,1,1.0, 1.0,1.0,1.0, 1.0,1,1.0, 1.0;
            MPC_solv.set_weight(1e-6, L_diag, K_diag);

            Eigen::VectorXd pos_des = kinDynSolver.integrateDIY(RobotState.q, RobotState.wbc_delta_q_final);
            RobotState.motors_pos_des = eigen2std(pos_des.block(7, 0, robot_nv - 6, 1));
            RobotState.motors_vel_des = eigen2std(RobotState.wbc_dq_final);
            RobotState.motors_tor_des = eigen2std(RobotState.wbc_tauJointRes);
        }

        pvtCtr.dataBusRead(RobotState);
        if (simTime <= openLoopCtrTime)
            pvtCtr.calMotorsPVT(110.0 / 1000.0 / 180.0 * 3.1415);
        else
        {
            pvtCtr.setJointPD(400, 15, "J_hip_l_roll");
            pvtCtr.setJointPD(200, 10, "J_hip_l_yaw");
            pvtCtr.setJointPD(300, 10, "J_hip_l_pitch");
            pvtCtr.setJointPD(300, 14, "J_knee_l_pitch");
            pvtCtr.setJointPD(300, 18, "J_ankle_l_pitch");
            pvtCtr.setJointPD(300, 16, "J_ankle_l_roll");
            pvtCtr.setJointPD(400, 15, "J_hip_r_roll");
            pvtCtr.setJointPD(200, 10, "J_hip_r_yaw");
            pvtCtr.setJointPD(300, 10, "J_hip_r_pitch");
            pvtCtr.setJointPD(300, 14, "J_knee_r_pitch");
            pvtCtr.setJointPD(300, 18, "J_ankle_r_pitch");
            pvtCtr.setJointPD(300, 16, "J_ankle_r_roll");
            pvtCtr.calMotorsPVT();
        }
        pvtCtr.dataBusWrite(RobotState);
        mj_interface.setMotorsTorque(RobotState.motors_tor_out);

        // stats
        if (simTime > openLoopCtrTime) {
            minZ = std::min(minZ, RobotState.base_pos(2));
            maxRoll = std::max(maxRoll, std::fabs(RobotState.base_rpy(0)));
            maxPitch = std::max(maxPitch, std::fabs(RobotState.base_rpy(1)));
        }

        if (getenv("DBG_CON") && simTime > openLoopCtrTime && RobotState.base_pos(0) > atof(getenv("DBG_CON")) &&
            RobotState.base_pos(0) < atof(getenv("DBG_CON")) + 0.05) {
            printf("# CONTACTS at t=%.2f x=%.2f:", simTime, RobotState.base_pos(0));
            for (int c=0;c<mj_data->ncon;c++){
                int g1=mj_data->contact[c].geom1, g2=mj_data->contact[c].geom2;
                const char*n1=mj_id2name(mj_model,mjOBJ_GEOM,g1);
                const char*n2=mj_id2name(mj_model,mjOBJ_GEOM,g2);
                printf(" [%s|%s z=%.3f]", n1?n1:"?", n2?n2:"?", mj_data->contact[c].pos[2]);
            }
            printf("\n");
        }
        if (simTime >= nextLog) {
            printf("%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                   simTime, RobotState.base_pos(0), RobotState.base_pos(1), RobotState.base_pos(2),
                   RobotState.base_rpy(0), RobotState.base_rpy(1), RobotState.base_rpy(2),
                   jsInterp.vx_L, jsInterp.wz_L,
                   RobotState.fe_l_pos_W(2), RobotState.fe_r_pos_W(2), mj_data->ncon);
            nextLog += 0.1;
        }

        if (!fellWarned && simTime > openLoopCtrTime && RobotState.base_pos(2) < 0.5) {
            printf("# FELL at t=%.2f z=%.3f\n", simTime, RobotState.base_pos(2));
            fellWarned = true;
        }
    }

    printf("# SUMMARY scenario=%s minZ=%.3f maxRoll=%.3f maxPitch=%.3f finalX=%.3f finalY=%.3f finalYaw=%.3f fell=%d\n",
           scenario.c_str(), minZ, maxRoll, maxPitch,
           RobotState.base_pos(0), RobotState.base_pos(1), RobotState.base_rpy(2), fellWarned ? 1 : 0);
    return 0;
}
