#pragma once
#include <xarm/wrapper/xarm_api.h>
<<<<<<< HEAD
/**
 * @brief move robot using constant speed to a target position
=======

XArmAPI* initRobot(const std::string &port, fp32 *initPose);
std::pair<fp32 *, fp32> moveOneStep(XArmAPI* arm, fp32 *initPose, fp32 *translation, fp32 newSpeed);
/**
 * @brief 
 * 
 *
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
 * 
 * @param port 
 * @param translation 
 * @param newSpeed
 * @return true 
 * @return false 
 */
bool moveRobot(const std::string &port, fp32 *translation, fp32 newSpeed);
<<<<<<< HEAD
=======

>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
