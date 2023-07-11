#pragma once
#include <xarm/wrapper/xarm_api.h>
/**
 * @brief move robot using constant speed to a target position
 * 
 * @param port 
 * @param translation 
 * @param newSpeed
 * @return true 
 * @return false 
 */
bool moveRobot(const std::string &port, fp32 *translation, fp32 newSpeed);
