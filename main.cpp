/**
 * @file main.cpp
 * @author kexin wei (wkx611@outlook.com)
 * @brief
 * a demo to test robot basic function like connection, motion control and etc.
 * @version 0.1
 * @date 2023-05-11
 *
 * SPDX-FileCopyrightText: 2023 Kristin Kexin Wei <wkx611@outlook.com>
 *
 * SPDX-License-Identifier: MIT
 * SPDX-License-Identifier: MIT License
 *
 */

#include <iostream>
#include <xarm/wrapper/xarm_api.h>
#include <test_api.h>
#include <control_traj.h>
// TODO2: move robot following a trajectory
// TODO3: move robot using velocity control following a trajectory (sin curve) and write the speed and acceleration to file
int main()
{
    const std::string port("192.168.1.154");
    // testApi(port);
    fp32 translation[6] = {0, 0, 5, 0, 2, 0};
    moveRobot(port, translation);
    return 1;
}
