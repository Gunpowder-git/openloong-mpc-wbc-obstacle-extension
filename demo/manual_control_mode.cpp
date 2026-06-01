/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024-2025 Humanoid Robot (Shanghai) Co., Ltd.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/

#include <mujoco/mujoco.h>
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include "GLFW_callbacks.h"
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
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "StateEst.h"

const double dt = 0.001;
const double dt_200Hz = 0.005;
// MuJoCo load and compile model
char error[1000] = "Could not load binary model";
mjModel *mj_model = mj_loadXML("../models/scene_manual_improved.xml", 0, error, 1000);
mjData *mj_data = mj_makeData(mj_model);


namespace {

enum class ExtraAction
{
    None,
    Jump,
    Bow,
    Crouch,
    StepUp,
    StepDown,
    FallToSit,
    GetUpFromBack,
    GetUpFromFront,
    WaveHand
};

bool keyPressedOnce(GLFWwindow *window, int key, bool &last)
{
    const bool now = glfwGetKey(window, key) == GLFW_PRESS;
    const bool edge = now && !last;
    last = now;
    return edge;
}

double clampValue(double value, double lo, double hi)
{
    return std::max(lo, std::min(value, hi));
}

double smooth01(double x)
{
    x = clampValue(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

} // namespace

int main(int argc, char **argv)
{
    // initialize classes
    UIctr uiController(mj_model, mj_data);                                             // UI control for Mujoco
    MJ_Interface mj_interface(mj_model, mj_data);                                      // data interface for Mujoco
    Pin_KinDyn kinDynSolver("../models/AzureLoong.urdf");                              // kinematics and dynamics solver
    DataBus RobotState(kinDynSolver.model_nv);                                         // data bus
    WBC_priority WBC_solv(kinDynSolver.model_nv, 18, 22, 0.7, mj_model->opt.timestep); // WBC solver
    MPC MPC_solv(dt_200Hz);                                                            // mpc controller
    GaitScheduler gaitScheduler(0.4, mj_model->opt.timestep);                          // gait scheduler
    PVT_Ctr pvtCtr(mj_model->opt.timestep, "../common/joint_ctrl_config.json");        // PVT joint control
    FootPlacement footPlacement;                                                       // foot-placement planner
    JoyStickInterpreter jsInterp(mj_model->opt.timestep);                              // desired baselink velocity generator
    DataLogger logger("../record/datalog.log");                                        // data logger
    StateEst StateModule(0.001);

    // initialize UI: GLFW
    uiController.iniGLFW();
    uiController.enableTracking();            // enable viewpoint tracking of the body 1 of the robot
    uiController.createWindow("Manual Control Mode", false); // NOTE: if the saveVideo is set to true, the raw recorded file could be 2.5 GB for 15 seconds!
    UIctr::ButtonState buttonState;

    // initialize variables
    double stand_legLength = 1.01; //-0.95; // desired baselink height
    double foot_height = 0.07;     // distance between the foot ankel joint and the bottom
    double xv_des = 0.6;           // desired velocity in x direction (moderate for controllable terrain walking)

    const int robot_nq = kinDynSolver.model_nv + 1;
    const int robot_nv = robot_nq - 1;

    RobotState.width_hips = 0.229;
    footPlacement.kp_vx = 0.03;
    footPlacement.kp_vy = 0.03;
    footPlacement.kp_wz = 0.03;
    footPlacement.stepHeight = 0.12;
    footPlacement.legLength = stand_legLength;

    const double originalStepHeight = footPlacement.stepHeight; // keep the original walking parameter unchanged
    double manualStepHeight = originalStepHeight;
    double manualTerrainHeightOffset = 0.0;
    double manualTerrainHeightTarget = 0.0;
    double extraBodyHeightOffset = 0.0;
    double extraPitchOffset = 0.0;
    ExtraAction extraAction = ExtraAction::None;
    double extraActionStartTime = -1.0;
    bool keyLatchB = false;
    bool keyLatchR = false;
    bool keyLatchF = false;
    bool keyLatchT = false;
    bool keyLatchG = false;
    bool keyLatchV = false;
    bool keyLatchM = false;
    bool keyLatchU = false;
    bool keyLatchJ = false;
    bool keyLatchC = false;
    bool keyLatchShift = false;
    bool recoveryAnnounced = false;
    bool fallenBackward = false;
    bool fallenForward = false;
    bool runMode = false;

    // Wave hand parameters
    double wavePhase = 0.0;

    std::vector<double> motors_pos_des(robot_nv - 6, 0);
    std::vector<double> motors_pos_cur(robot_nv - 6, 0);
    std::vector<double> motors_vel_des(robot_nv - 6, 0);
    std::vector<double> motors_vel_cur(robot_nv - 6, 0);
    std::vector<double> motors_tau_des(robot_nv - 6, 0);
    std::vector<double> motors_tau_cur(robot_nv - 6, 0);

    // ini position and posture for foot-end and hand
    Eigen::Vector3d fe_l_pos_L_des = {-0.018, 0.113, -stand_legLength};
    Eigen::Vector3d fe_r_pos_L_des = {-0.018, -0.116, -stand_legLength};
    Eigen::Vector3d fe_l_eul_L_des = {-0.000, -0.008, -0.000};
    Eigen::Vector3d fe_r_eul_L_des = {0.000, -0.008, 0.000};
    Eigen::Matrix3d fe_l_rot_des = eul2Rot(fe_l_eul_L_des(0), fe_l_eul_L_des(1), fe_l_eul_L_des(2));
    Eigen::Matrix3d fe_r_rot_des = eul2Rot(fe_r_eul_L_des(0), fe_r_eul_L_des(1), fe_r_eul_L_des(2));

    Eigen::VectorXd hd_l_des, hd_r_des;
    hd_l_des.resize(7);
    hd_r_des.resize(7);

    hd_l_des << 0.475, -1.12, 1.9, 0.86, -0.356, 0, 0;
    hd_r_des << -0.475, -1.12, -1.9, 0.86, 0.356, 0, 0;

    // Initialize arm wave target
    Eigen::VectorXd armJointsWaveTarget(14);
    armJointsWaveTarget.block(0, 0, 7, 1) = hd_l_des;
    armJointsWaveTarget.block(7, 0, 7, 1) = hd_r_des;

    auto resLeg = kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des, fe_r_rot_des, fe_r_pos_L_des);
    Eigen::VectorXd qIniDes = Eigen::VectorXd::Zero(mj_model->nq, 1);
    qIniDes.block(7, 0, mj_model->nq - 7, 1) = resLeg.jointPosRes;
    qIniDes.block(7, 0, 7, 1) = hd_l_des;
    qIniDes.block(14, 0, 7, 1) = hd_r_des;
    WBC_solv.setQini(qIniDes, RobotState.q);

    // register variable name for data logger
    logger.addIterm("dyn_time", 1);
    logger.addIterm("motors_pos_cur", robot_nv - 6);
    logger.addIterm("motors_pos_des", robot_nv - 6);
    logger.addIterm("motors_tor_cur", robot_nv - 6);
    logger.addIterm("motors_tor_des", robot_nv - 6);
    logger.addIterm("motors_tor_out", robot_nv - 6);
    logger.addIterm("motors_vel_des", robot_nv - 6);
    logger.addIterm("motors_vel_cur", robot_nv - 6);
    logger.addIterm("FL_est", 3);
    logger.addIterm("FR_est", 3);
    logger.addIterm("wbc_FrRes", 12);
    logger.addIterm("base_pos_des", 3);
    logger.addIterm("base_pos", 3);
    logger.addIterm("base_pos_est", 3);
    logger.addIterm("baseLinVel", 3);
    logger.addIterm("base_vel_est", 3);
    logger.addIterm("base_rpy", 3);
    logger.addIterm("eul_est", 3);
	logger.addIterm("Ufe", 13);
	logger.addIterm("legState", 1);
	logger.addIterm("motionState", 1);

    logger.finishItermAdding();

    //// -------------------------- main loop --------------------------------

    int MPC_count = 0; // count for controlling the mpc running period

    double openLoopCtrTime = 3;
    double startSteppingTime = 7;
    double startWalkingTime = 10;
    double simEndTime = 200;

    mjtNum simstart = mj_data->time;
    double simTime = mj_data->time;

    while (!glfwWindowShouldClose(uiController.window))
    {
        simstart = mj_data->time;
        while (mj_data->time - simstart < 1.0 / 60.0 && uiController.runSim)
        {
            mj_step(mj_model, mj_data);
            simTime = mj_data->time;
            // Read the sensors:
            mj_interface.updateSensorValues();
            mj_interface.dataBusWrite(RobotState);

            if (simTime > 1 && StateModule.flag_init)
            {
                std::cout << "init state module" << std::endl;
                StateModule.init(RobotState);
            }

            // input from joystick
            // space: start and stop stepping (after 3s)
            // w: forward walking
            // s: stop forward walking
            // a: turning left
            // d: turning right
            buttonState = uiController.getButtonState();
            if (simTime > openLoopCtrTime)
            {
                if (buttonState.key_space && RobotState.motionState == DataBus::Stand)
                {
					gaitScheduler.start();
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                    RobotState.motionState = DataBus::Walk;
                }
                else if (buttonState.key_space && RobotState.motionState == DataBus::Walk && fabs(jsInterp.vxLGen.y) < 0.01)
                {
                    RobotState.motionState = DataBus::Walk2Stand;
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                }

                // Is the forward key currently held? Used so that A/D alone = clean turn-in-place
                // (no unintended forward creep), while W+A / W+D still curve-walks.
                const bool wHeld = glfwGetKey(uiController.window, GLFW_KEY_W) == GLFW_PRESS;

                if (buttonState.key_a && RobotState.motionState != DataBus::Stand)
                {
                    if (jsInterp.wzLGen.yDes < 0)
                        jsInterp.setWzDesLPara(0, 0.4);
                    else
                        jsInterp.setWzDesLPara(0.3, 0.6); // snappier turn response
                    if (!wHeld)
                        jsInterp.setVxDesLPara(0.0, 0.4); // turn in place: cancel forward motion
                }
                if (buttonState.key_d && RobotState.motionState != DataBus::Stand)
                {
                    if (jsInterp.wzLGen.yDes > 0)
                        jsInterp.setWzDesLPara(0, 0.4);
                    else
                        jsInterp.setWzDesLPara(-0.3, 0.6); // snappier turn response
                    if (!wHeld)
                        jsInterp.setVxDesLPara(0.0, 0.4); // turn in place: cancel forward motion
                }

                if (buttonState.key_w && RobotState.motionState != DataBus::Stand)
                {
                    // Check if Shift is pressed for running
                    if (glfwGetKey(uiController.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                        glfwGetKey(uiController.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
                    {
                        jsInterp.setVxDesLPara(0.9, 1.4); // Run speed (faster response)
                        runMode = true;
                    }
                    else
                    {
                        jsInterp.setVxDesLPara(xv_des, 1.2); // Normal walk speed (faster response)
                        runMode = false;
                    }
                }

                if (buttonState.key_s && RobotState.motionState != DataBus::Stand)
                    jsInterp.setVxDesLPara(0, 0.4);

                if (buttonState.key_h)
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));


                // -------- Added manual actions. Existing W/A/S/D/SPACE/H code above is copied unchanged. --------
                const bool robotUpright = std::fabs(RobotState.base_rpy(0)) < 0.35 &&
                                          std::fabs(RobotState.base_rpy(1)) < 0.35 &&
                                          RobotState.base_pos(2) > 0.70;

                if (buttonState.key_b && RobotState.motionState != DataBus::Stand)
                {
                    // New backward command; original forward/stop/turn commands are not changed.
                    jsInterp.setVxDesLPara(-0.25, 2.0);
                }

                if (buttonState.key_r)
                {
                    manualStepHeight = clampValue(manualStepHeight + 0.01, originalStepHeight, 0.20);
                    std::cout << "manual step height = " << manualStepHeight << " m" << std::endl;
                }

                if (buttonState.key_f)
                {
                    manualStepHeight = clampValue(manualStepHeight - 0.01, originalStepHeight, 0.20);
                    std::cout << "manual step height = " << manualStepHeight << " m" << std::endl;
                }

                // Jump action (J key) - relaxed conditions
                if (buttonState.key_j && extraAction == ExtraAction::None)
                {
                    extraAction = ExtraAction::Jump;
                    extraActionStartTime = simTime;
                    std::cout << "start jump" << std::endl;
                }

                // Crouch/Bow action (C key) - relaxed conditions
                if (buttonState.key_c && extraAction == ExtraAction::None)
                {
                    extraAction = ExtraAction::Crouch;
                    extraActionStartTime = simTime;
                    std::cout << "start crouch" << std::endl;
                }

                if (buttonState.key_v && extraAction == ExtraAction::None)
                {
                    extraAction = ExtraAction::Bow;
                    extraActionStartTime = simTime;
                    std::cout << "start bow" << std::endl;
                }

                if (buttonState.key_t && extraAction == ExtraAction::None)
                {
                    extraAction = ExtraAction::StepUp;
                    extraActionStartTime = simTime;
                    manualTerrainHeightTarget = clampValue(manualTerrainHeightTarget + manualStepHeight, -0.10, 0.45);
                    std::cout << "step up target terrain offset = " << manualTerrainHeightTarget << " m" << std::endl;
                }

                if (buttonState.key_g && extraAction == ExtraAction::None)
                {
                    extraAction = ExtraAction::StepDown;
                    extraActionStartTime = simTime;
                    manualTerrainHeightTarget = clampValue(manualTerrainHeightTarget - manualStepHeight, -0.10, 0.45);
                    std::cout << "step down target terrain offset = " << manualTerrainHeightTarget << " m" << std::endl;
                }

                // Wave hand action (M key) - relaxed conditions
                if (buttonState.key_m && extraAction == ExtraAction::None)
                {
                    extraAction = ExtraAction::WaveHand;
                    extraActionStartTime = simTime;
                    wavePhase = 0.0;
                    std::cout << "start wave hand" << std::endl;
                }

                // Manual balance-recovery trigger (U key): start stepping to recover balance.
                // (A grounded robot cannot be stood up by this controller; see fall handling below.)
                if (buttonState.key_u && extraAction == ExtraAction::None &&
                    RobotState.motionState == DataBus::Stand && RobotState.base_pos(2) > 0.70)
                {
                    gaitScheduler.start();
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                    jsInterp.setVxDesLPara(0.0, 0.3);
                    jsInterp.setWzDesLPara(0.0, 0.3);
                    RobotState.motionState = DataBus::Walk;
                    std::cout << "Manual balance recovery: stepping in place" << std::endl;
                }
            }

            StateModule.set(RobotState);
            StateModule.update();
            StateModule.get(RobotState);

            // ---------------- Balance recovery / honest fall handling ----------------
            // Two tiers:
            //  (1) Developing fall (moderate tilt, base still up): automatically start
            //      stepping so the foot-placement capture terms can catch the balance.
            //      This is genuine recovery-by-stepping for disturbances while standing.
            //  (2) Full fall (robot on the ground): this MPC+WBC controller is a two-foot
            //      contact stand/walk controller and physically cannot push the body back up
            //      from a grounded posture (verified). We report this honestly instead of
            //      running a get-up animation that does not actually lift the robot.
            if (simTime > openLoopCtrTime + 1.0 && extraAction == ExtraAction::None)
            {
                const double absRoll  = std::fabs(RobotState.base_rpy(0));
                const double absPitch = std::fabs(RobotState.base_rpy(1));
                const bool fullyDown = RobotState.base_pos(2) < 0.55 || absRoll > 1.2 || absPitch > 1.2;
                const bool developingFall = !fullyDown && (absRoll > 0.22 || absPitch > 0.22) &&
                                            RobotState.base_pos(2) > 0.70;

                if (fullyDown)
                {
                    if (!recoveryAnnounced)
                    {
                        std::cout << "[recovery] Robot has fallen to the ground. This controller (two-foot-contact "
                                     "MPC+WBC) cannot autonomously stand up from a grounded posture. "
                                     "Press H then SPACE to resume once the robot is upright." << std::endl;
                        recoveryAnnounced = true;
                    }
                    RobotState.motionState = DataBus::Stand;
                }
                else if (developingFall && RobotState.motionState == DataBus::Stand)
                {
                    // recovery stepping: begin stepping in place to catch balance
                    gaitScheduler.start();
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                    jsInterp.setVxDesLPara(0.0, 0.3);
                    jsInterp.setWzDesLPara(0.0, 0.3);
                    RobotState.motionState = DataBus::Walk;
                    recoveryAnnounced = false;
                    std::cout << "[recovery] disturbance detected - stepping to recover balance" << std::endl;
                }
                else if (!fullyDown)
                {
                    recoveryAnnounced = false;
                }
            }

            // update kinematics and dynamics info
            kinDynSolver.dataBusRead(RobotState);
            kinDynSolver.computeJ_dJ();
            kinDynSolver.computeDyn();
            kinDynSolver.dataBusWrite(RobotState);

            // update F EST
            StateModule.setF(RobotState);
            StateModule.updateF();
            StateModule.getF(RobotState);

            if (simTime >= openLoopCtrTime && simTime < openLoopCtrTime + 0.002)
            {
                RobotState.motionState = DataBus::Stand;
            }

            if (RobotState.motionState == DataBus::Walk2Stand || simTime <= openLoopCtrTime)
                jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));

            // switch between walk and stand
            if (RobotState.motionState == DataBus::Walk || RobotState.motionState == DataBus::Walk2Stand)
            {
                jsInterp.step();
                RobotState.js_pos_des(2) = stand_legLength + foot_height; // pos z is not assigned in jyInterp
                jsInterp.dataBusWrite(RobotState);                        // only pos x, pos y, theta z, vel x, vel y , omega z are rewrote.

                MPC_solv.enable();

                // gait scheduler
                gaitScheduler.dataBusRead(RobotState);
                gaitScheduler.step();
                gaitScheduler.dataBusWrite(RobotState);

                footPlacement.dataBusRead(RobotState);
                footPlacement.getSwingPos();
                footPlacement.dataBusWrite(RobotState);
            }
			else{
				MPC_solv.disable();
			}

            if (simTime <= openLoopCtrTime || RobotState.motionState == DataBus::Walk2Stand)
            {
                WBC_solv.setQini(qIniDes, RobotState.q);
                WBC_solv.fe_l_pos_des_W = RobotState.fe_l_pos_W;
                WBC_solv.fe_r_pos_des_W = RobotState.fe_r_pos_W;
                WBC_solv.fe_l_rot_des_W = RobotState.fe_l_rot_W;
                WBC_solv.fe_r_rot_des_W = RobotState.fe_r_rot_W;
                WBC_solv.pCoMDes = RobotState.pCoM_W;
            }

			// ------------- MPC ------------
			MPC_count = MPC_count + 1;
			if (MPC_count > (dt_200Hz / dt - 1)) { // MPC_count = 1, 2, 3, 4, 5(5 run MPC)
				MPC_solv.dataBusRead(RobotState);
				MPC_solv.cal();

				MPC_count = 0;
			}

			if (RobotState.motionState==DataBus::Walk || RobotState.motionState==DataBus::Walk2Stand) {
				MPC_solv.dataBusWrite(RobotState);
			}
			else {
				RobotState.Fr_ff = Eigen::VectorXd::Zero(12);
				RobotState.des_ddq = Eigen::VectorXd::Zero(RobotState.model_nv);
				RobotState.des_dq = Eigen::VectorXd::Zero(RobotState.model_nv);
				RobotState.des_delta_q = Eigen::VectorXd::Zero(RobotState.model_nv);
				RobotState.base_rpy_des << 0.0, 0.0, jsInterp.thetaZ;
				RobotState.base_pos_des= RobotState.js_pos_des;
				RobotState.base_pos_des(2) = stand_legLength+foot_height;// - 0.005;
				// printf("base_rpy_des: %.4f, %.4f, %.4f\n", data.base_rpy_des[0], data.base_rpy_des[1], data.base_rpy_des[2]);
				// printf("base_pos_des: %.4f, %.4f, %.4f\n", data.base_pos_des[0], data.base_pos_des[1], data.base_pos_des[2]);
				RobotState.Fr_ff<<0,0,370,0,0,0,
						0,0,370,0,0,0;
			}


            // -------- Added action parameter layer. When no new action is active, all offsets are exactly zero. --------
            manualTerrainHeightOffset = Ramp(manualTerrainHeightOffset, manualTerrainHeightTarget, 0.12 * mj_model->opt.timestep);
            extraBodyHeightOffset = 0.0;
            extraPitchOffset = 0.0;
            footPlacement.stepHeight = originalStepHeight;

            // Reset arm wave target to default - use block assignment
            armJointsWaveTarget.block(0, 0, 7, 1) = hd_l_des;
            armJointsWaveTarget.block(7, 0, 7, 1) = hd_r_des;

            if (extraAction != ExtraAction::None)
            {
                const double actionTime = simTime - extraActionStartTime;

                if (extraAction == ExtraAction::Jump)
                {
                    // Real jump with actual upward force
                    if (actionTime < 0.30)
                    {
                        // Deep crouch preparation
                        extraBodyHeightOffset = -0.20 * smooth01(actionTime / 0.30);
                    }
                    else if (actionTime < 0.45)
                    {
                        // Explosive push-off with upward force
                        const double pushPhase = (actionTime - 0.30) / 0.15;
                        extraBodyHeightOffset = -0.20 + 0.40 * smooth01(pushPhase);

                        // Apply strong upward force during push-off
                        if (pushPhase > 0.3) {
                            double upwardForce = 1200.0 * smooth01((pushPhase - 0.3) / 0.7);
                            RobotState.Fr_ff(2) += upwardForce;  // Left foot Z
                            RobotState.Fr_ff(8) += upwardForce;  // Right foot Z
                        }
                    }
                    else if (actionTime < 0.75)
                    {
                        // Flight phase - zero contact
                        const double flightPhase = (actionTime - 0.45) / 0.30;
                        extraBodyHeightOffset = 0.20 * (1.0 - 4.0 * (flightPhase - 0.5) * (flightPhase - 0.5));

                        // Zero all contact forces during flight
                        RobotState.Fr_ff.setZero();
                    }
                    else if (actionTime < 1.05)
                    {
                        // Landing absorption
                        const double landPhase = (actionTime - 0.75) / 0.30;
                        extraBodyHeightOffset = 0.20 * (1.0 - smooth01(landPhase)) - 0.15 * smooth01(landPhase);
                    }
                    else if (actionTime < 1.45)
                    {
                        // Recovery to standing
                        const double recoverPhase = (actionTime - 1.05) / 0.40;
                        extraBodyHeightOffset = -0.15 * (1.0 - smooth01(recoverPhase));
                    }
                    else
                    {
                        extraAction = ExtraAction::None;
                        std::cout << "jump completed" << std::endl;
                    }
                }
                else if (extraAction == ExtraAction::Crouch)
                {
                    // Deep crouch/bow action
                    const double crouchDuration = 2.5;
                    if (actionTime < 0.8)
                    {
                        const double s = smooth01(actionTime / 0.8);
                        extraBodyHeightOffset = -0.25 * s;
                        extraPitchOffset = 0.25 * s;
                    }
                    else if (actionTime < crouchDuration - 0.8)
                    {
                        // Hold crouch position
                        extraBodyHeightOffset = -0.25;
                        extraPitchOffset = 0.25;
                    }
                    else if (actionTime < crouchDuration)
                    {
                        const double s = 1.0 - smooth01((actionTime - (crouchDuration - 0.8)) / 0.8);
                        extraBodyHeightOffset = -0.25 * s;
                        extraPitchOffset = 0.25 * s;
                    }
                    else
                    {
                        extraAction = ExtraAction::None;
                        std::cout << "crouch completed" << std::endl;
                    }
                }
                else if (extraAction == ExtraAction::Bow)
                {
                    if (actionTime < 1.10)
                    {
                        const double s = smooth01(actionTime / 1.10);
                        extraPitchOffset = 0.18 * s;
                        extraBodyHeightOffset = -0.015 * s;
                    }
                    else if (actionTime < 2.20)
                    {
                        const double s = 1.0 - smooth01((actionTime - 1.10) / 1.10);
                        extraPitchOffset = 0.18 * s;
                        extraBodyHeightOffset = -0.015 * s;
                    }
                    else
                    {
                        extraAction = ExtraAction::None;
                    }
                }
                else if (extraAction == ExtraAction::StepUp || extraAction == ExtraAction::StepDown)
                {
                    // Improved step control with better weight shift
                    footPlacement.stepHeight = std::max(originalStepHeight, manualStepHeight + 0.05);

                    // Add slight forward lean for better stability on stairs
                    if (extraAction == ExtraAction::StepDown) {
                        extraPitchOffset = 0.05 * smooth01(clampValue(actionTime / 1.0, 0.0, 1.0));
                    }

                    if (actionTime > 2.50)
                    {
                        extraAction = ExtraAction::None;
                    }
                }
                else if (extraAction == ExtraAction::WaveHand)
                {
                    // Wave hand motion - right arm waves
                    const double waveDuration = 3.0;
                    const double waveFreq = 2.0; // Hz

                    if (actionTime < waveDuration)
                    {
                        wavePhase = actionTime * waveFreq * 2.0 * 3.141592653589793;

                        // Right arm wave motion
                        armJointsWaveTarget(7) = -0.475 + 0.75 * 0.0; // shoulder roll - lift arm
                        armJointsWaveTarget(8) = -1.12 - 0.8;  // shoulder pitch - raise arm
                        armJointsWaveTarget(9) = -1.9 + 0.3 * std::sin(wavePhase); // shoulder yaw - wave motion
                        armJointsWaveTarget(10) = 0.86 + 0.5; // elbow pitch - bend elbow
                        armJointsWaveTarget(11) = 0.356 + 0.4 * std::sin(wavePhase); // wrist roll - wave

                        // Keep left arm in default position
                        armJointsWaveTarget.block(0, 0, 7, 1) = hd_l_des;
                    }
                    else
                    {
                        extraAction = ExtraAction::None;
                        std::cout << "wave hand completed" << std::endl;
                    }
                }
                else if (extraAction == ExtraAction::GetUpFromBack)
                {
                    // Get up from supine (lying on back) position - improved sequence
                    if (!recoveryAnnounced)
                    {
                        std::cout << "Get-up from back sequence started (8 phases)" << std::endl;
                        recoveryAnnounced = true;
                    }

                    if (actionTime < 1.0)
                    {
                        // Phase 1: Stabilize and assess
                        extraBodyHeightOffset = 0.0;
                        extraPitchOffset = 0.0;
                    }
                    else if (actionTime < 2.5)
                    {
                        // Phase 2: Bend knees and prepare to roll
                        const double s = smooth01((actionTime - 1.0) / 1.5);
                        extraBodyHeightOffset = -0.20 * s;
                        extraPitchOffset = -0.10 * s;
                    }
                    else if (actionTime < 4.0)
                    {
                        // Phase 3: Roll to side
                        const double s = smooth01((actionTime - 2.5) / 1.5);
                        extraBodyHeightOffset = -0.20;
                        extraPitchOffset = -0.10 + 0.15 * s;
                    }
                    else if (actionTime < 5.5)
                    {
                        // Phase 4: Push up to hands and knees
                        const double s = smooth01((actionTime - 4.0) / 1.5);
                        extraBodyHeightOffset = -0.20 + 0.05 * s;
                        extraPitchOffset = 0.05 + 0.10 * s;
                    }
                    else if (actionTime < 7.0)
                    {
                        // Phase 5: Rise to kneeling position
                        const double s = smooth01((actionTime - 5.5) / 1.5);
                        extraBodyHeightOffset = -0.15 + 0.05 * s;
                        extraPitchOffset = 0.15 * (1.0 - s);
                    }
                    else if (actionTime < 9.0)
                    {
                        // Phase 6: Transition to standing
                        const double s = smooth01((actionTime - 7.0) / 2.0);
                        extraBodyHeightOffset = -0.10 * (1.0 - s);
                        extraPitchOffset = 0.0;
                    }
                    else
                    {
                        extraAction = ExtraAction::None;
                        fallenBackward = false;
                        std::cout << "Get-up from back completed - press H to reset position" << std::endl;
                    }
                }
                else if (extraAction == ExtraAction::GetUpFromFront)
                {
                    // Get up from prone (lying face down) position - improved sequence
                    if (!recoveryAnnounced)
                    {
                        std::cout << "Get-up from front sequence started (8 phases)" << std::endl;
                        recoveryAnnounced = true;
                    }

                    if (actionTime < 1.0)
                    {
                        // Phase 1: Stabilize
                        extraBodyHeightOffset = 0.0;
                        extraPitchOffset = 0.0;
                    }
                    else if (actionTime < 2.5)
                    {
                        // Phase 2: Push up with arms
                        const double s = smooth01((actionTime - 1.0) / 1.5);
                        extraBodyHeightOffset = -0.25 * s;
                        extraPitchOffset = 0.15 * s;
                    }
                    else if (actionTime < 4.0)
                    {
                        // Phase 3: Bring knees under body
                        const double s = smooth01((actionTime - 2.5) / 1.5);
                        extraBodyHeightOffset = -0.25 + 0.05 * s;
                        extraPitchOffset = 0.15;
                    }
                    else if (actionTime < 5.5)
                    {
                        // Phase 4: Rise to hands and knees
                        const double s = smooth01((actionTime - 4.0) / 1.5);
                        extraBodyHeightOffset = -0.20 + 0.05 * s;
                        extraPitchOffset = 0.15 - 0.05 * s;
                    }
                    else if (actionTime < 7.0)
                    {
                        // Phase 5: Rise to kneeling
                        const double s = smooth01((actionTime - 5.5) / 1.5);
                        extraBodyHeightOffset = -0.15 + 0.05 * s;
                        extraPitchOffset = 0.10 * (1.0 - s);
                    }
                    else if (actionTime < 9.0)
                    {
                        // Phase 6: Stand up
                        const double s = smooth01((actionTime - 7.0) / 2.0);
                        extraBodyHeightOffset = -0.10 * (1.0 - s);
                    }
                    else
                    {
                        extraAction = ExtraAction::None;
                        fallenForward = false;
                        std::cout << "Get-up from front completed - press H to reset position" << std::endl;
                    }
                }
                else if (extraAction == ExtraAction::FallToSit)
                {
                    if (!recoveryAnnounced)
                    {
                        std::cout << "fallen posture detected: gradually lowering to a sitting-height command; press H after stable to reuse the original standing routine" << std::endl;
                        recoveryAnnounced = true;
                    }
                    manualTerrainHeightTarget = 0.0;
                    manualTerrainHeightOffset = Ramp(manualTerrainHeightOffset, 0.0, 0.08 * mj_model->opt.timestep);
                    extraBodyHeightOffset = -0.34 * smooth01(clampValue(actionTime / 3.5, 0.0, 1.0));
                    extraPitchOffset = 0.10 * smooth01(clampValue(actionTime / 3.5, 0.0, 1.0));
                    if (actionTime > 3.8)
                    {
                        extraAction = ExtraAction::None;
                    }
                }
            }

            RobotState.base_pos_des(2) = clampValue(RobotState.base_pos_des(2) + manualTerrainHeightOffset + extraBodyHeightOffset,
                                                    0.72, stand_legLength + foot_height + 0.50);
            RobotState.base_rpy_des(1) = clampValue(RobotState.base_rpy_des(1) + extraPitchOffset, -0.24, 0.24);

            // Apply arm wave targets if wave action is active
            if (extraAction == ExtraAction::WaveHand)
            {
                qIniDes.block(7, 0, 14, 1) = armJointsWaveTarget;
            }
            else
            {
                // Reset to default arm position
                qIniDes.block(7, 0, 7, 1) = hd_l_des;
                qIniDes.block(14, 0, 7, 1) = hd_r_des;
            }

            // ------------- WBC ------------
            // WBC Calculation
            WBC_solv.dataBusRead(RobotState);
            WBC_solv.computeDdq(kinDynSolver);
            WBC_solv.computeTau();
            WBC_solv.dataBusWrite(RobotState);

            // get the final joint command
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
				// Verified-stable MPC weights (same as the proven walk_mpc_wbc_joystick baseline).
				L_diag << 1.0, 1.0, 1.0,    // eul
						1e-3, 100.0, 1.0,   // pCoM
						1e-3, 1e-3, 1e-3,   // w
						1.0, 100.0, 1.0;    // vCoM
				K_diag << 1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						1.0;
				MPC_solv.set_weight(1e-6, L_diag, K_diag);

                Eigen::VectorXd pos_des = kinDynSolver.integrateDIY(RobotState.q, RobotState.wbc_delta_q_final);
                RobotState.motors_pos_des = eigen2std(pos_des.block(7, 0, robot_nv - 6, 1));
                RobotState.motors_vel_des = eigen2std(RobotState.wbc_dq_final);
                RobotState.motors_tor_des = eigen2std(RobotState.wbc_tauJointRes);
            }

            // joint PVT controller - improved gains for better response
            pvtCtr.dataBusRead(RobotState);
            if (simTime <= openLoopCtrTime)
            {
                pvtCtr.calMotorsPVT(110.0 / 1000.0 / 180.0 * 3.1415);
            }
            else
            {
                double kp = 1.0;
                double kd = 1.0;

                // Verified-stable PD gains (same as the proven walk_mpc_wbc_joystick baseline).
                pvtCtr.setJointPD(400 * kp, 15 * kd, "J_hip_l_roll");
                pvtCtr.setJointPD(200 * kp, 10 * kd, "J_hip_l_yaw");
                pvtCtr.setJointPD(300 * kp, 10 * kd, "J_hip_l_pitch");
                pvtCtr.setJointPD(300 * kp, 14 * kd, "J_knee_l_pitch");
                pvtCtr.setJointPD(300 * kp, 18 * kd, "J_ankle_l_pitch");
                pvtCtr.setJointPD(300 * kp, 16 * kd, "J_ankle_l_roll");

                pvtCtr.setJointPD(400 * kp, 15 * kd, "J_hip_r_roll");
                pvtCtr.setJointPD(200 * kp, 10 * kd, "J_hip_r_yaw");
                pvtCtr.setJointPD(300 * kp, 10 * kd, "J_hip_r_pitch");
                pvtCtr.setJointPD(300 * kp, 14 * kd, "J_knee_r_pitch");
                pvtCtr.setJointPD(300 * kp, 18 * kd, "J_ankle_r_pitch");
                pvtCtr.setJointPD(300 * kp, 16 * kd, "J_ankle_r_roll");
                pvtCtr.calMotorsPVT();
            }
            pvtCtr.dataBusWrite(RobotState);

            // give the joint torque command to Webots
            mj_interface.setMotorsTorque(RobotState.motors_tor_out);

            logger.startNewLine();
            logger.recItermData("dyn_time", simTime);
            logger.recItermData("motors_pos_cur", RobotState.motors_pos_cur);
            logger.recItermData("motors_pos_des", RobotState.motors_pos_des);
            logger.recItermData("motors_tor_cur", RobotState.motors_tor_cur);
            logger.recItermData("motors_tor_des", RobotState.motors_tor_des);
            logger.recItermData("motors_tor_out", RobotState.motors_tor_out);
            logger.recItermData("motors_vel_cur", RobotState.motors_vel_cur);
            logger.recItermData("motors_vel_des", RobotState.motors_vel_des);
            logger.recItermData("FL_est", RobotState.FL_est);
            logger.recItermData("FR_est", RobotState.FR_est);
            logger.recItermData("wbc_FrRes", RobotState.wbc_FrRes);
            logger.recItermData("base_pos_des", RobotState.base_pos_des);
            logger.recItermData("base_pos", RobotState.base_pos);
            logger.recItermData("FL_est", RobotState.FL_est);
            logger.recItermData("FR_est", RobotState.FR_est);
            logger.recItermData("wbc_FrRes", RobotState.wbc_FrRes);
            logger.recItermData("base_pos_des", RobotState.base_pos_des);
            logger.recItermData("base_pos", RobotState.base_pos);
            logger.recItermData("base_pos_est", RobotState.base_pos_est);
            logger.recItermData("baseLinVel", RobotState.baseLinVel);
            logger.recItermData("base_vel_est", RobotState.base_vel_est);
            logger.recItermData("base_rpy", RobotState.base_rpy);
            logger.recItermData("eul_est", RobotState.eul_est);
			logger.recItermData("Ufe", RobotState.fe_react_tau_cmd);
			logger.recItermData("legState", RobotState.legState);
			logger.recItermData("motionState", RobotState.motionState);
            logger.finishLine();
        }

        if (mj_data->time >= simEndTime)
            break;

        uiController.updateScene();
    };
    // free visualization storage
    uiController.Close();

    return 0;
}
