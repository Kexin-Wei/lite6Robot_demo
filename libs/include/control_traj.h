#pragma once
#include <xarm/wrapper/xarm_api.h>
/**
 * @brief move robot using constant speed to a target position //FIXME
 * 
 * @param port 
 * @param translation 
 * @return true 
 * @return false 
 */
bool moveRobot(const std::string &port, fp32 *translation);
