#include <control_traj.h>
<<<<<<< HEAD
bool moveRobot(const std::string &port, fp32 *translation, fp32 newSpeed)
=======

XArmAPI* initRobot(const std::string &port, fp32 *initPose)
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
{
    XArmAPI *arm = new XArmAPI(port);
    sleep_milliseconds(500);
    if (arm->error_code != 0)
        arm->clean_error();
    if (arm->warn_code != 0)
        arm->clean_warn();
    arm->motion_enable(true);
    arm->set_mode(0);
    arm->set_state(0);
    sleep_milliseconds(500);

    printf("=========================================\n");
<<<<<<< HEAD
    fp32 initPose[6];
=======
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
    int ret;
    ret = arm->get_position(initPose);
    printf("current position is, ret=%d, pose=[%f, %f, %f, %f, %f, %f]\nPress Enter to continue ...",
           ret, initPose[0], initPose[1], initPose[2], initPose[3], initPose[4], initPose[5]);
    std::cin.get();

<<<<<<< HEAD
=======
    return arm;
}

std::pair<fp32 *, fp32> moveOneStep(XArmAPI* arm, fp32 *initPose, fp32 *translation, fp32 newSpeed)
{
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
    fp32 targetPose[6];
    for (int i = 0; i < 6; i++)
    {
        targetPose[i] = initPose[i] + translation[i];
    }
<<<<<<< HEAD

    fp32 lastSpeed{arm->last_used_tcp_speed};
    printf("Last used speed is %f\n", lastSpeed);
=======
    fp32 lastSpeed = arm->last_used_tcp_speed;
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
    if (abs(newSpeed) < abs(lastSpeed))
    {
        lastSpeed = newSpeed;
    }
<<<<<<< HEAD
    printf("Moving speed is set to %f\n", lastSpeed);

    // record steady_clock
    auto start = std::chrono::steady_clock::now();
    ret = arm->set_position(targetPose, -1, lastSpeed, 0, 0, true, 20, false, 0);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("New position is, ret=%d, pose=[%f, %f, %f, %f, %f, %f], cost time %ld\nPress Enter to continue ...",
           ret, targetPose[0], targetPose[1], targetPose[2], targetPose[3], targetPose[4], targetPose[5], duration.count());

    printf("finished.");
    return 1;
}
=======
    int ret;
    ret = arm->set_position(targetPose, -1, lastSpeed, 0, 0, true, 20, false, 0);
    std::pair<fp32 *, fp32> newPosNewSpeed{targetPose,lastSpeed};
    return newPosNewSpeed;
}

bool moveRobot(const std::string &port, fp32 *translation, fp32 newSpeed)
{
    fp32 initPose[6];
    XArmAPI *arm = initRobot(port, initPose);

    printf("Last used speed is %f\n", arm->last_used_tcp_speed);

    // record steady_clock
    auto start = std::chrono::steady_clock::now();
    auto newPoseNewSpeed = moveOneStep(arm, initPose, translation, newSpeed);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto targetPose = newPoseNewSpeed.first;
    auto endSpeed = newPoseNewSpeed.second;

    printf("Moving speed is set to %f\n", endSpeed);
    printf("New position is, pose=[%f, %f, %f, %f, %f, %f], cost time %ld\nPress Enter to continue ...",
           targetPose[0], targetPose[1], targetPose[2], targetPose[3], targetPose[4], targetPose[5], duration.count());

    printf("finished.");
    return 1;
}

>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
