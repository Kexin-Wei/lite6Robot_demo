#pragma once
#include <xarm/wrapper/xarm_api.h>

XArmAPI* initRobot(const std::string &port, fp32 *initPose);
std::pair<fp32 *, fp32> moveOneStep(XArmAPI* arm, fp32 *initPose, fp32 *translation, fp32 newSpeed);
/**
 * @brief 
 * 
 *
 * 
 * @param port 
 * @param translation 
 * @param newSpeed
 * @return true 
 * @return false 
 */
bool moveRobot(const std::string &port, fp32 *translation, fp32 newSpeed);

