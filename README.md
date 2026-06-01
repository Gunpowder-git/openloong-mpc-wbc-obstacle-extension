# OpenLoong MPC-WBC Obstacle Avoidance Extension

This repository contains course-project modifications and additional demos based on the open-source **OpenLoong-Dyn-Control** project.

Original upstream project:

* Repository: `loongOpen/OpenLoong-Dyn-Control`
* Original project: OpenLoong Dynamics Control
* Original license: Apache License 2.0

This repository is **not an official OpenLoong repository** and is **not a standalone full distribution** of OpenLoong-Dyn-Control. It is an educational extension that demonstrates humanoid locomotion, manual control, virtual LiDAR-based obstacle detection, and simple automatic obstacle avoidance in MuJoCo simulation.

## Project Overview

The project extends the original MPC-WBC humanoid robot control framework with additional simulation scenarios and control modes.

Main additions include:

* Enhanced automatic obstacle avoidance mode
* Virtual LiDAR scanning using MuJoCo ray casting
* Low-obstacle step-over strategy
* High-obstacle bypass strategy
* Arc-length style waypoint tracking
* Enhanced MuJoCo scenes with stairs, slopes, low obstacles, high obstacles, and goal markers
* Manual multi-function keyboard control mode
* Foot-ground contact parameter tuning for improved simulation stability
* Fall detection and simulation reset logic for testing
* Additional demo programs for validation and debugging

The goal of this project is to demonstrate how an MPC-WBC humanoid locomotion framework can be extended from basic walking to simple perception-planning-control simulation tasks.

## Relationship to OpenLoong-Dyn-Control

This project is based on OpenLoong-Dyn-Control.

The original project provides a humanoid robot motion control framework based on:

* MPC: Model Predictive Control
* WBC: Whole-Body Control
* MuJoCo simulation
* Pinocchio dynamics computation
* qpOASES quadratic programming

Our work focuses on secondary development and engineering integration. We do not claim that the original OpenLoong-Dyn-Control framework was written by us.

## Main Features

### 1. Manual Control Mode

The manual control mode supports keyboard-based robot operation, including:

* Start / stop stepping
* Forward walking
* Backward walking
* Left and right turning
* Step height adjustment
* Squat / bow motion
* Arm waving
* Stair-related motion tests
* Recovery trigger tests

This mode is used to validate the basic MPC-WBC locomotion pipeline and additional action interfaces.

### 2. Enhanced Automatic Obstacle Avoidance

The automatic mode adds a simple perception-planning-control loop:

1. The simulation reads obstacle geometry from the MuJoCo scene.
2. A virtual LiDAR module uses MuJoCo ray casting to estimate front clearance and terrain height.
3. Low obstacles are classified as step-over targets.
4. High obstacles are treated as bypass targets.
5. The planner outputs desired forward velocity, yaw rate, and step height.
6. The original gait scheduler, foot placement, MPC, WBC, and PVT control pipeline executes the motion.

### 3. Low Obstacle Step-over

Low obstacles are handled by increasing the swing foot height. The step height is estimated from obstacle height and virtual LiDAR terrain information, with an additional clearance margin and an upper bound for stability.

### 4. High Obstacle Bypass

High obstacles are not crossed directly. Instead, the planner generates side waypoints around the obstacle. This avoids forcing the swing leg to exceed a safe height range.

### 5. Improved Contact Stability

The project includes modifications to foot-ground contact settings in MuJoCo model files. In particular, foot collision geometry parameters are adjusted to improve consistency between the controller's friction assumptions and MuJoCo's simulated contact behavior.

### 6. Fall Detection and Test Reset

The enhanced demo includes fall detection based on base roll, pitch, and height thresholds.

Important note: this is **not a real physical stand-up controller**. The reset logic is only a simulation utility for testing. A real fallen humanoid recovery controller would require multi-contact planning involving hands, knees, torso contact, and full-body motion sequencing.

## Recommended Repository Usage

This repository is intended to be used together with the original OpenLoong-Dyn-Control codebase.

Recommended workflow:

1. Clone the original OpenLoong-Dyn-Control repository.
2. Copy the modified files from this repository into the corresponding folders.
3. Build the project using CMake.
4. Run the additional demo targets.

Example:

```bash
git clone https://github.com/loongOpen/OpenLoong-Dyn-Control.git
cd OpenLoong-Dyn-Control

# Copy this repository's modified files into the same folder structure.
# Then build:

mkdir build
cd build
cmake ..
make
```

Example demo targets may include:

```bash
./manual_control_mode
./auto_obstacle_mode
./enhanced_auto_mode
./recovery_test
./headless_test
```

The exact target names depend on the provided `CMakeLists.txt`.

## Suggested File Structure

```text
.
├── README.md
├── LICENSE
├── NOTICE
├── DISCLAIMER.md
├── THIRD_PARTY_NOTICES.md
├── CHANGES.md
├── CMakeLists.txt
├── demo/
│   ├── enhanced_auto_mode.cpp
│   ├── auto_obstacle_mode.cpp
│   ├── manual_control_mode.cpp
│   ├── recovery_test.cpp
│   ├── headless_test.cpp
│   └── snapshot.cpp
├── models/
│   ├── AzureLoong.xml
│   ├── scene_enhanced.xml
│   ├── scene_auto_obstacle.xml
│   ├── scene_manual_improved.xml
│   ├── scene_manual_multi_env.xml
│   ├── scene_complete_environment.xml
│   └── meshes/
│       └── ramp_wedge.obj
├── algorithm/
│   └── foot_placement.cpp
└── sim_interface/
    ├── GLFW_callbacks.cpp
    └── GLFW_callbacks.h
```

Only modified or newly added files should be included. Unmodified upstream files should generally be obtained from the original OpenLoong-Dyn-Control repository.

## What We Modified

The main modifications include:

* Added new MuJoCo simulation scenes for obstacle avoidance and multi-environment testing.
* Added automatic obstacle avoidance demo logic.
* Added virtual LiDAR ray-casting logic.
* Added obstacle classification based on object height.
* Added waypoint-based bypass logic for high obstacles.
* Added dynamic swing foot height adjustment for low obstacles.
* Added manual control extensions.
* Tuned foot-ground contact parameters for more stable MuJoCo simulation.
* Added fall detection and simulation reset logic.
* Updated build configuration for new demo targets.

For more details, see `CHANGES.md`.

## Limitations

This project has several limitations:

* The perception module is simulation-based, not a real hardware sensor pipeline.
* The virtual LiDAR is implemented using MuJoCo ray casting.
* The obstacle avoidance logic is a simple rule-based planner, not a full SLAM or global navigation system.
* The fall recovery logic is only simulation reset / light recovery testing, not real physical stand-up control.
* The project has not been validated on real hardware.
* The control parameters are tuned for the provided MuJoCo simulation scenes and may not transfer directly to physical robots.

## License

This project follows the Apache License 2.0, consistent with the upstream OpenLoong-Dyn-Control project.

Original OpenLoong-Dyn-Control copyright and attribution notices are preserved where applicable.

Modified files should include clear notices stating that they were modified for this course project.

## Citation / Acknowledgement

This project is based on OpenLoong-Dyn-Control.

If you use this repository, please also cite or acknowledge the original OpenLoong-Dyn-Control project:

```bibtex
@software{Robot2024OpenLoong,
  author = {Humanoid Robot (Shanghai) Co., Ltd},
  title = {{OpenLoong-DynamicsControl: Motion control framework of humanoid robot based on MPC and WBC}},
  url = {https://github.com/loongOpen/OpenLoong-Dyn-Control.git},
  year = {2024}
}
```

## Disclaimer

This repository is for educational and research demonstration purposes only. It is not an official OpenLoong release and is not intended for direct deployment on physical robots without additional validation, safety checks, and hardware-specific adaptation.
