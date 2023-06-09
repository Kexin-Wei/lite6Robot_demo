/**
 * Software License Agreement (MIT License)
 * 
 * @copyright Copyright (c) 2022, UFACTORY, Inc.
 * 
 * All rights reserved.
 * 
 * @author Zhang <jimy92@163.com>
 * @author Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>
 */

#ifndef CORE_INSTRUCTION_UXBUS_CMD_TCP_H_
#define CORE_INSTRUCTION_UXBUS_CMD_TCP_H_

#include "xarm/core/instruction/uxbus_cmd.h"
#include "xarm/core/port/socket.h"

class UxbusCmdTcp : public UxbusCmd {
public:
  UxbusCmdTcp(SocketPort *arm_port);
  ~UxbusCmdTcp(void);

  int check_xbus_prot(unsigned char *datas, int funcode);
  int send_pend(int funcode, int num, int timeout, unsigned char *ret_data);
  int send_xbus(int funcode, unsigned char *datas, int num);
  void close(void);
  int is_ok(void);

  int get_prot_flag(void);
  int set_prot_flag(int prot_flag = 2);

private:
  SocketPort *arm_port_;
  int bus_flag_;
  int prot_flag_;
  int TX2_PROT_CON_ = 2;         // tcp cmd prot
  int TX2_PROT_HEAT_ = 1;        // tcp heat prot
  int TX2_BUS_FLAG_MIN_ = 1;     // the min cmd num
  int TX2_BUS_FLAG_MAX_ = 5000;  // the max cmd num
};

#endif
