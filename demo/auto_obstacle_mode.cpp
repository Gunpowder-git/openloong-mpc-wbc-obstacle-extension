/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024-2025 Humanoid Robot (Shanghai) Co., Ltd.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>

Automatic obstacle-avoidance mode.
 - Perception: obstacles are read directly from the live MuJoCo model (every geom whose
   name begins with "obs_"). Each is classified as STEP-OVER (low) or GO-AROUND (tall).
 - Planning: a trajectory-rollout planner picks a (vx, wz) that maximises progress toward the
   goal while keeping clearance from the tall obstacles; low obstacles trigger a higher step.
 - Control: reuses the proven gait / joystick / MPC / WBC walking pipeline unchanged.

Set the environment variable HEADLESS=1 to run without a GLFW window (prints the base
trajectory to stdout) for automated verification.
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
#include <cstdlib>
#include <limits>
#include <vector>
#include "StateEst.h"

const double dt = 0.001;
const double dt_200Hz = 0.005;
// MuJoCo load and compile model
char error[1000] = "Could not load binary model";
mjModel *mj_model = mj_loadXML("../models/scene_auto_obstacle.xml", 0, error, 1000);
mjData *mj_data = mj_makeData(mj_model);


namespace {

struct Obstacle
{
    double x;
    double y;
    double radius;
    double top;       // world height of the top surface
    bool   steppable; // low enough to step over
};

struct RolloutCommand
{
    double vx{0.0};
    double wz{0.0};
    double stepHeight{0.12};
    bool reachedGoal{false};
};

double clampValue(double value, double lo, double hi)
{
    return std::max(lo, std::min(value, hi));
}

double wrapToPi(double a)
{
    while (a > 3.141592653589793) a -= 2.0 * 3.141592653589793;
    while (a < -3.141592653589793) a += 2.0 * 3.141592653589793;
    return a;
}

// Read every "obs_*" geom from the model and classify it.  Worldbody geom_pos is already
// in world coordinates, so this is a faithful perception of the static obstacle field.
std::vector<Obstacle> perceiveObstacles(const mjModel *m, double stepOverMaxTop)
{
    std::vector<Obstacle> obs;
    for (int g = 0; g < m->ngeom; ++g)
    {
        const char *name = mj_id2name(m, mjOBJ_GEOM, g);
        if (!name || std::string(name).rfind("obs_", 0) != 0)
            continue;
        const double px = m->geom_pos[3 * g + 0];
        const double py = m->geom_pos[3 * g + 1];
        const double pz = m->geom_pos[3 * g + 2];
        const double sx = m->geom_size[3 * g + 0];
        const double sy = m->geom_size[3 * g + 1];
        const double sz = m->geom_size[3 * g + 2];
        Obstacle o;
        o.x = px;
        o.y = py;
        if (m->geom_type[g] == mjGEOM_CYLINDER || m->geom_type[g] == mjGEOM_SPHERE)
        {
            o.radius = sx;
            o.top = pz + (m->geom_type[g] == mjGEOM_SPHERE ? sx : sy);
        }
        else // box (and others): use horizontal half-extents
        {
            o.radius = std::max(sx, sy);
            o.top = pz + sz;
        }
        o.steppable = (o.top <= stepOverMaxTop);
        obs.push_back(o);
    }
    return obs;
}

struct Waypoint { double x; double y; };

// Predictive path planning: because the obstacle field is static and fully perceived up
// front, we precompute a safe polyline that detours around each tall obstacle at a generous
// lateral offset, then follow it with pure pursuit. This is far more reliable for a balance-
// limited biped than reactive steering (no late, destabilising hard turns).
std::vector<Waypoint> buildPath(double startX, double startY,
                                double goalX, double goalY,
                                std::vector<Obstacle> obstacles)
{
    // tall obstacles only, sorted by x (travel direction)
    std::vector<Obstacle> tall;
    for (const auto &o : obstacles) if (!o.steppable) tall.push_back(o);
    std::sort(tall.begin(), tall.end(), [](const Obstacle &a, const Obstacle &b){ return a.x < b.x; });

    std::vector<Waypoint> wps;
    wps.push_back({startX, startY});
    double curY = startY;
    for (const auto &o : tall)
    {
        // keep the robot CENTRE this far from the obstacle centre (pillar r + body + margin)
        const double keep = o.radius + 0.85;
        // choose the side (relative to obstacle) that requires the smaller lateral move,
        // defaulting to +y; offset the obstacle's own y.
        const double sideSign = (curY >= o.y) ? 1.0 : -1.0;
        const double passY = o.y + sideSign * keep;
        // approach and departure waypoints so the detour is a smooth arc, not a kink
        wps.push_back({o.x - keep, passY});
        wps.push_back({o.x + keep * 0.3, passY});
        curY = passY;
    }
    wps.push_back({goalX, goalY});
    return wps;
}

RolloutCommand purePursuit(double x, double y, double yaw,
                           const std::vector<Waypoint> &wps, int &wpIdx,
                           double goalX, double goalY,
                           const std::vector<Obstacle> &obstacles,
                           double nominalStepHeight)
{
    RolloutCommand cmd;
    const double goalDist = std::hypot(goalX - x, goalY - y);
    if (goalDist < 0.45)
    {
        cmd.reachedGoal = true;
        return cmd;
    }

    // advance through reached waypoints (keep the last/goal one)
    while (wpIdx < (int)wps.size() - 1 &&
           std::hypot(wps[wpIdx].x - x, wps[wpIdx].y - y) < 0.55)
        wpIdx++;

    const double tx = wps[wpIdx].x, ty = wps[wpIdx].y;
    const double desiredHeading = std::atan2(ty - y, tx - x);
    const double headErr = wrapToPi(desiredHeading - yaw);

    cmd.wz = clampValue(1.2 * headErr, -0.40, 0.40);
    double v = 0.32 * clampValue(1.0 - std::fabs(headErr) / 0.8, 0.18, 1.0);
    cmd.vx = clampValue(v, 0.0, 0.32);

    // step-over handling for low obstacles directly ahead
    cmd.stepHeight = nominalStepHeight;
    for (const auto &o : obstacles)
    {
        if (!o.steppable) continue;
        const double dx = o.x - x, dy = o.y - y;
        const double forward = std::cos(yaw) * dx + std::sin(yaw) * dy;
        const double lateral = std::fabs(-std::sin(yaw) * dx + std::cos(yaw) * dy);
        if (forward > 0.0 && forward < 1.1 && lateral < 0.7)
        {
            cmd.stepHeight = std::max(cmd.stepHeight, o.top + 0.09);
            cmd.vx = std::min(cmd.vx, 0.26);
        }
    }
    cmd.stepHeight = clampValue(cmd.stepHeight, nominalStepHeight, 0.22);
    return cmd;
}

} // namespace

int main(int argc, char **argv)
{
    const bool headless = std::getenv("HEADLESS") != nullptr;
    const double simEndTime = std::getenv("SIM_END") ? atof(std::getenv("SIM_END")) : 200.0;

    // initialize classes
    UIctr uiController(mj_model, mj_data);
    MJ_Interface mj_interface(mj_model, mj_data);
    Pin_KinDyn kinDynSolver("../models/AzureLoong.urdf");
    DataBus RobotState(kinDynSolver.model_nv);
    WBC_priority WBC_solv(kinDynSolver.model_nv, 18, 22, 0.7, mj_model->opt.timestep);
    MPC MPC_solv(dt_200Hz);
    GaitScheduler gaitScheduler(0.4, mj_model->opt.timestep);
    PVT_Ctr pvtCtr(mj_model->opt.timestep, "../common/joint_ctrl_config.json");
    FootPlacement footPlacement;
    JoyStickInterpreter jsInterp(mj_model->opt.timestep);
    DataLogger logger("../record/datalog.log");
    StateEst StateModule(0.001);

    if (!headless)
    {
        uiController.iniGLFW();
        uiController.enableTracking();
        uiController.createWindow("Auto Obstacle Mode", false);
    }
    UIctr::ButtonState buttonState;

    // initialize variables
    double stand_legLength = 1.01;
    double foot_height = 0.07;
    double xv_des = 0.6;

    const int robot_nq = kinDynSolver.model_nv + 1;
    const int robot_nv = robot_nq - 1;

    RobotState.width_hips = 0.229;
    footPlacement.kp_vx = 0.03;
    footPlacement.kp_vy = 0.03;
    footPlacement.kp_wz = 0.03;
    footPlacement.stepHeight = 0.12;
    footPlacement.legLength = stand_legLength;

    const double originalStepHeight = footPlacement.stepHeight;

    // ---- PERCEPTION: read obstacles from the live model ----
    const double stepOverMaxTop = 0.10;
    std::vector<Obstacle> obstacles = perceiveObstacles(mj_model, stepOverMaxTop);
    double goalX = 10.5, goalY = 0.0;
    {
        int gid = mj_name2id(mj_model, mjOBJ_GEOM, "goal_marker");
        if (gid >= 0) { goalX = mj_model->geom_pos[3 * gid + 0]; goalY = mj_model->geom_pos[3 * gid + 1]; }
    }
    std::cout << "[perception] " << obstacles.size() << " obstacles detected; goal at ("
              << goalX << ", " << goalY << ")" << std::endl;
    for (const auto &o : obstacles)
        std::cout << "   obs x=" << o.x << " y=" << o.y << " r=" << o.radius
                  << " top=" << o.top << (o.steppable ? " [step-over]" : " [go-around]") << std::endl;

    // ---- PLANNING: precompute a safe path around the tall obstacles ----
    std::vector<Waypoint> path = buildPath(0.0, 0.0, goalX, goalY, obstacles);
    int wpIdx = 0;
    std::cout << "[planning] path waypoints:";
    for (const auto &w : path) std::cout << " (" << w.x << "," << w.y << ")";
    std::cout << std::endl;

    double lastAutoPlanTime = -10.0;
    double autoVxCmd = 0.0;
    double autoWzCmd = 0.0;
    double autoStepHeightCmd = originalStepHeight;
    bool   goalReached = false;
    double goalReachedTime = -1.0;

    std::vector<double> motors_pos_des(robot_nv - 6, 0);
    std::vector<double> motors_vel_des(robot_nv - 6, 0);
    std::vector<double> motors_tau_des(robot_nv - 6, 0);

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

    auto resLeg = kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des, fe_r_rot_des, fe_r_pos_L_des);
    Eigen::VectorXd qIniDes = Eigen::VectorXd::Zero(mj_model->nq, 1);
    qIniDes.block(7, 0, mj_model->nq - 7, 1) = resLeg.jointPosRes;
    qIniDes.block(7, 0, 7, 1) = hd_l_des;
    qIniDes.block(14, 0, 7, 1) = hd_r_des;
    WBC_solv.setQini(qIniDes, RobotState.q);

    logger.addIterm("dyn_time", 1);
    logger.addIterm("base_pos", 3);
    logger.addIterm("base_rpy", 3);
    logger.finishItermAdding();

    int MPC_count = 0;
    double openLoopCtrTime = 3;

    double simTime = mj_data->time;
    mjtNum simstart = mj_data->time;
    double nextPrint = 0.0;

    while (headless ? (mj_data->time < simEndTime) : !glfwWindowShouldClose(uiController.window))
    {
        simstart = mj_data->time;
        const bool runInner = headless ? true : uiController.runSim;
        while (mj_data->time - simstart < 1.0 / 60.0 && runInner)
        {
            mj_step(mj_model, mj_data);
            simTime = mj_data->time;
            mj_interface.updateSensorValues();
            mj_interface.dataBusWrite(RobotState);

            if (simTime > 1 && StateModule.flag_init)
                StateModule.init(RobotState);

            if (!headless)
                buttonState = uiController.getButtonState();

            // -------- Automatic obstacle navigation --------
            if (simTime > openLoopCtrTime + 0.40)
            {
                if (RobotState.motionState == DataBus::Stand && !goalReached)
                {
                    gaitScheduler.start();
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                    RobotState.motionState = DataBus::Walk;
                }

                if (RobotState.motionState != DataBus::Stand && simTime - lastAutoPlanTime > 0.20)
                {
                    // Robot world pose from the simulator's free joint (stands in for an
                    // onboard localization system; the walking controller itself does not
                    // measure absolute position).
                    const double rx = mj_data->qpos[0];
                    const double ry = mj_data->qpos[1];
                    const RolloutCommand c = purePursuit(rx, ry,
                                                         RobotState.base_rpy(2), path, wpIdx,
                                                         goalX, goalY, obstacles, originalStepHeight);
                    autoVxCmd = c.vx;
                    autoWzCmd = c.wz;
                    autoStepHeightCmd = c.stepHeight;
                    lastAutoPlanTime = simTime;

                    if (c.reachedGoal && !goalReached)
                    {
                        goalReached = true;
                        goalReachedTime = simTime;
                        std::cout << "[navigation] GOAL REACHED at x=" << rx
                                  << " y=" << ry << " t=" << simTime << std::endl;
                    }

                    jsInterp.setVxDesLPara(autoVxCmd, 1.0);
                    jsInterp.setWzDesLPara(autoWzCmd, 0.8);
                }

                // stop and stand a moment after reaching the goal
                if (goalReached && RobotState.motionState == DataBus::Walk &&
                    fabs(jsInterp.vxLGen.y) < 0.02 && simTime - goalReachedTime > 1.5)
                {
                    RobotState.motionState = DataBus::Walk2Stand;
                    jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
                }
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
                footPlacement.stepHeight = autoStepHeightCmd;

                gaitScheduler.dataBusRead(RobotState);
                gaitScheduler.step();
                gaitScheduler.dataBusWrite(RobotState);

                footPlacement.dataBusRead(RobotState);
                footPlacement.getSwingPos();
                footPlacement.dataBusWrite(RobotState);
            }
            else
            {
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
            if (MPC_count > (dt_200Hz / dt - 1))
            {
                MPC_solv.dataBusRead(RobotState);
                MPC_solv.cal();
                MPC_count = 0;
            }

            if (RobotState.motionState == DataBus::Walk || RobotState.motionState == DataBus::Walk2Stand)
            {
                MPC_solv.dataBusWrite(RobotState);
            }
            else
            {
                RobotState.Fr_ff = Eigen::VectorXd::Zero(12);
                RobotState.des_ddq = Eigen::VectorXd::Zero(RobotState.model_nv);
                RobotState.des_dq = Eigen::VectorXd::Zero(RobotState.model_nv);
                RobotState.des_delta_q = Eigen::VectorXd::Zero(RobotState.model_nv);
                RobotState.base_rpy_des << 0.0, 0.0, jsInterp.thetaZ;
                RobotState.base_pos_des = RobotState.js_pos_des;
                RobotState.base_pos_des(2) = stand_legLength + foot_height;
                RobotState.Fr_ff << 0, 0, 370, 0, 0, 0, 0, 0, 370, 0, 0, 0;
            }

            // ------------- WBC ------------
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
                K_diag << 1.0, 1.0, 1.0, 1.0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 1.0, 1.0;
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

            logger.startNewLine();
            logger.recItermData("dyn_time", simTime);
            logger.recItermData("base_pos", RobotState.base_pos);
            logger.recItermData("base_rpy", RobotState.base_rpy);
            logger.finishLine();

            if (headless && simTime >= nextPrint)
            {
                printf("t=%.2f x=%.3f y=%.3f z=%.3f yaw=%.3f roll=%.3f pitch=%.3f goalDist=%.3f wp=%d tgt=(%.2f,%.2f) vx=%.2f wz=%.2f\n",
                       simTime, mj_data->qpos[0], mj_data->qpos[1], RobotState.base_pos(2),
                       RobotState.base_rpy(2), RobotState.base_rpy(0), RobotState.base_rpy(1),
                       std::hypot(goalX - mj_data->qpos[0], goalY - mj_data->qpos[1]),
                       wpIdx, path[wpIdx].x, path[wpIdx].y, autoVxCmd, autoWzCmd);
                nextPrint += 0.5;
            }
        }

        if (mj_data->time >= simEndTime)
            break;
        if (headless && goalReached && simTime - goalReachedTime > 3.0)
            break;

        if (!headless)
            uiController.updateScene();
    }

    if (headless)
    {
        const double gd = std::hypot(goalX - mj_data->qpos[0], goalY - mj_data->qpos[1]);
        printf("# RESULT reachedGoal=%d finalX=%.3f finalY=%.3f goalDist=%.3f z=%.3f fell=%d\n",
               goalReached ? 1 : 0, mj_data->qpos[0], mj_data->qpos[1], gd,
               RobotState.base_pos(2), (RobotState.base_pos(2) < 0.5) ? 1 : 0);
    }
    else
    {
        uiController.Close();
    }

    return 0;
}
