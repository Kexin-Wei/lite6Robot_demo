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
#include <xarm/wrapper/xarm_api.h>
#include <test_api.h>
#include <control_traj.h>

#define RUN_CODE 2

#include <fstream>
// TODO3: move robot using velocity control following a trajectory (sin curve) and write the speed and acceleration to file
int main()
{
    const std::string port("192.168.1.154");
#if RUN_CODE == 0
    testApi(port);

#elif RUN_CODE == 1
    // 1. move robot using position control
    fp32 translation[6] = {0, 5, 5, 0, 0, 0};
    fp32 newSpeed{1};
    moveRobot(port, translation, newSpeed);

#elif RUN_CODE == 2
    // TODO2: move robot following a trajectory, which wroten in the file
    // read trajectory from file trajectory.txt
    std::ifstream trajFile("trajectory.txt", std::ios::in);
    if (!trajFile.is_open())
    {
        std::cout << "Error opening file trajectory.txt" << std::endl;
        return 0;
    }
    std::vector<std::vector<float>> traj;
    std::string line;
    while (std::getline(trajFile, line))
    {
        std::istringstream iss(line);
        std::vector<float> point;
        float value;
        while (iss >> value)
        {
            point.push_back(value);
        }
        traj.push_back(point);
    }
    for (auto point : traj)
    {
        for (auto value : point)
        {
            printf("%f ", value);
        }
        printf("\n");
    }
#endif
    return 1;
}
