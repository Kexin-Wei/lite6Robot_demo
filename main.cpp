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

// TODO2: move robot following a trajectory
// TODO3: move robot using velocity control following a trajectory (sin curve) and write the speed and acceleration to file
int main()
{
    // testApi();
    // TODO1: move robot using constant speed to a target position
    std::string port("192.168.1.100");

    XArmAPI *arm = new XArmAPI(port);
    sleep_milliseconds(500);
    if (arm->error_code != 0) arm->clean_error();
    if (arm->warn_code != 0) arm->clean_warn();
    arm->motion_enable(true);
    arm->set_mode(0);
    arm->set_state(0);
    sleep_milliseconds(500);

    printf("=========================================\n");
    fp32 initPose[6];
    int ret;
    ret = arm->get_position(initPose);
    printf("current position is, ret=%d, pose=[%f, %f, %f, %f, %f, %f]\nPress Enter to continue ...", 
        ret, initPose[0], initPose[1], initPose[2], initPose[3], initPose[4], initPose[5]);
    std::cin.get();

    fp32 translation[6] = {30, 0, 20, 0, 0, 0};
    fp32 targetPose[6];
    for (int i = 0; i < 6; i++)
    {
        targetPose[i] = initPose[i] + translation[i];
    }
    ret = arm->set_position(targetPose, true);
    printf("new position is, ret=%d, pose=[%f, %f, %f, %f, %f, %f]\nPress Enter to continue ...", 
        ret, targetPose[0], targetPose[1], targetPose[2], targetPose[3], targetPose[4], targetPose[5]);

    printf("finished.");

}
