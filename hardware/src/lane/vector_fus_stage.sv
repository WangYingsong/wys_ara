// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author:  Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Description:
// This is Ara's vector execution stage. This contains the functional units
// of each lane, namely the ALU and the Multiplier/FPU.

module vector_fus_stage import ara_pkg::*; import rvv_pkg::*; import cf_math_pkg::idx_width; #(
    parameter  int           unsigned NrLanes      = 0,
    // Support for floating-point data types
    parameter  fpu_support_e          FPUSupport   = FPUSupportHalfSingleDouble,
    // External support for vfrec7, vfrsqrt7
    parameter  fpext_support_e        FPExtSupport = FPExtSupportEnable,
    // Support for fixed-point data types
    parameter  fixpt_support_e        FixPtSupport = FixedPointEnable,
    // Type used to address vector register file elements
    parameter  type                   vaddr_t      = logic,
    // Dependant parameters. DO NOT CHANGE!
    localparam int           unsigned DataWidth    = $bits(elen_simd_t), // wys
    localparam type                   strb_t       = logic [DataWidth/8-1:0]
  ) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic [idx_width(NrLanes)-1:0]     lane_id_i,
    // Interface with Dispatcher
    output logic                              vxsat_flag_o,
    input  vxrm_t                             alu_vxrm_i,
    // Interface with CVA6
    output logic           [4:0]              fflags_ex_o,
    output logic                              fflags_ex_valid_o,
    // Interface with the lane sequencer
    input  vfu_operation_t                    vfu_operation_i,
    input  logic                              vfu_operation_valid_i,
    output logic                              alu_ready_o,
    output logic           [NrVInsn-1:0]      alu_vinsn_done_o,
    output logic                              mfpu_ready_o,
    output logic           [NrVInsn-1:0]      mfpu_vinsn_done_o,
    // Interface with the operand queues
    input  elen_simd_t     [1:0]              alu_operand_i, // wys
    input  logic           [1:0]              alu_operand_valid_i,
    output logic           [1:0]              alu_operand_ready_o,
    input  elen_simd_t     [2:0]              mfpu_operand_i, // wys
    input  logic           [2:0]              mfpu_operand_valid_i,
    output logic           [2:0]              mfpu_operand_ready_o,
    // Interface with the vector register file
    output logic                              alu_result_req_o,
    output vid_t                              alu_result_id_o,
    output vaddr_t                            alu_result_addr_o,
    output elen_simd_t                        alu_result_wdata_o, // wys
    output strb_t                             alu_result_be_o,
    input  logic                              alu_result_gnt_i,
    // Multiplier/FPU
    output logic                              mfpu_result_req_o,
    output vid_t                              mfpu_result_id_o,
    output vaddr_t                            mfpu_result_addr_o,
    output elen_simd_t                        mfpu_result_wdata_o, // wys
    output strb_t                             mfpu_result_be_o,
    input  logic                              mfpu_result_gnt_i,
    // Interface with the Slide Unit
    input  elen_simd_t                        sldu_operand_i,
    output logic         [Nr_SIMD-1:0]        sldu_alu_req_valid_o,
    input  logic                              sldu_alu_valid_i,
    output logic                              sldu_alu_ready_o,
    input  logic                              sldu_alu_gnt_i,
    output logic         [Nr_SIMD-1:0]        sldu_mfpu_req_valid_o,
    input  logic                              sldu_mfpu_valid_i,
    output logic                              sldu_mfpu_ready_o,
    input  logic                              sldu_mfpu_gnt_i,
    // Interface with the Mask unit
    output elen_simd_t     [NrMaskFUnits-1:0] mask_operand_o,
    output logic           [NrMaskFUnits-1:0] mask_operand_valid_o,
    input  logic           [NrMaskFUnits-1:0] mask_operand_ready_i,
    input  strb_t                             mask_i,
    input  logic                              mask_valid_i,
    output logic                              mask_ready_o
  );

  ///////////////
  //  Signals  //
  ///////////////

  // If the mask unit has instruction queue depth > 1, change the following lines.
  // If we have concurrent masked MUL and ADD operations, mask_i and mask_valid_i are
  // erroneously broadcasted and accepted to/by both the units. The mask unit must tag its
  // broadcasted signals if more masked instructions can be in different units at the same time.
  logic alu_mask_ready;
  logic mfpu_mask_ready;
  assign mask_ready_o = alu_mask_ready | mfpu_mask_ready;

  // saturation selection
  logic alu_vxsat, mfpu_vxsat;
  assign vxsat_flag_o = mfpu_vxsat | alu_vxsat;

  elen_t [Nr_SIMD-1:0]      sldu_operand_i_;
  
  logic  [Nr_SIMD-1:0][7:0] mask_i_;
  
  for(genvar i=0;i<Nr_SIMD;i++) begin :gen_sldu_operand_i
    assign sldu_operand_i_[i] = sldu_operand_i[ELEN*(i+1)-1:ELEN*i];
    assign mask_i_[i]         = mask_i[8*(i+1)-1:8*i];
  end

  //////////////////
  //  Vector ALU  //
  //////////////////

  // wys
  logic   [Nr_SIMD-1:0]              alu_ready_o_;
  logic   [Nr_SIMD-1:0][NrVInsn-1:0] alu_vinsn_done_o_;
  logic   [Nr_SIMD-1:0][1:0]         alu_operand_ready_o_;
  logic   [Nr_SIMD-1:0]              alu_result_req_o_;
  vid_t   [Nr_SIMD-1:0]              alu_result_id_o_;
  vaddr_t [Nr_SIMD-1:0]              alu_result_addr_o_;

  assign alu_ready_o         = &alu_ready_o_;
  assign alu_result_req_o    = |alu_result_req_o_;
  assign alu_result_id_o     = alu_result_id_o_[0];
  assign alu_result_addr_o   = alu_result_addr_o_[0];

  for(genvar i=0;i<Nr_SIMD;i++) begin :gen_alu
    elen_t [1:0] alu_operand_i_; 
    elen_t       alu_result_wdata_o_;
    logic  [7:0] alu_result_be_o_;
    
    
    assign alu_operand_ready_o                     = (i==0)?alu_operand_ready_o_[i]:alu_operand_ready_o_[i] & alu_operand_ready_o;
    assign alu_vinsn_done_o                        = (i==0)?alu_vinsn_done_o_[i]:alu_vinsn_done_o_[i] & alu_vinsn_done_o;
    assign alu_operand_i_[0]                       = alu_operand_i[0][ELEN*(i+1)-1:ELEN*i];
    assign alu_operand_i_[1]                       = alu_operand_i[1][ELEN*(i+1)-1:ELEN*i];
    assign alu_result_wdata_o[ELEN*(i+1)-1:ELEN*i] = alu_result_wdata_o_;
    assign alu_result_be_o[8*(i+1)-1:8*i]          = alu_result_be_o_;

    logic sldu_alu_req_valid_o_;
    logic sldu_alu_ready_o_;

    assign sldu_alu_req_valid_o[i] = sldu_alu_req_valid_o_;
    assign sldu_alu_ready_o        = (i==0)?sldu_alu_ready_o_:sldu_alu_ready_o_ & sldu_alu_ready_o;

    elen_t mask_operand_o_alu;
    logic  mask_operand_valid_o_alu;
    logic  alu_mask_ready_alu;

    assign mask_operand_o[MaskFUAlu][ELEN*(i+1)-1:ELEN*i] = mask_operand_o_alu;
    assign mask_operand_valid_o[MaskFUAlu]                = (i==0)?mask_operand_valid_o_alu:mask_operand_valid_o_alu & mask_operand_valid_o[MaskFUAlu];
    assign alu_mask_ready                                 = (i==0)?alu_mask_ready_alu:alu_mask_ready_alu & alu_mask_ready; 

    valu #(
    .NrLanes(NrLanes),
    .FixPtSupport(FixPtSupport),
    .vaddr_t(vaddr_t)
    ) i_valu (
      .clk_i                (clk_i                                  ),
      .rst_ni               (rst_ni                                 ),
      .lane_id_i            (i                                      ),
      // Interface with Dispatcher
      .vxsat_flag_o         (alu_vxsat                              ),
      .alu_vxrm_i           (alu_vxrm_i                             ),
      // Interface with the lane sequencer
      .vfu_operation_i      (vfu_operation_i                        ),
      .vfu_operation_valid_i(vfu_operation_valid_i                  ),
      .alu_ready_o          (alu_ready_o_[i]                        ),
      .alu_vinsn_done_o     (alu_vinsn_done_o_[i]                   ),
      // Interface with the operand queues
      // wys
      .alu_operand_i        (alu_operand_i_                         ),
      .alu_operand_valid_i  (alu_operand_valid_i                    ),
      .alu_operand_ready_o  (alu_operand_ready_o_[i]                ),
      // Interface with the vector register file
      .alu_result_req_o     (alu_result_req_o_[i]                   ),
      .alu_result_addr_o    (alu_result_addr_o_[i]                  ),
      .alu_result_id_o      (alu_result_id_o_[i]                    ),
      // wys
      .alu_result_wdata_o   (alu_result_wdata_o_                    ),
      .alu_result_be_o      (alu_result_be_o_                       ),
      .alu_result_gnt_i     (alu_result_gnt_i & alu_result_req_o_[i]),
      // Interface with the Slide Unit
      .alu_red_valid_o      (sldu_alu_req_valid_o_                  ),
      // wys
      .sldu_operand_i       (sldu_operand_i_[i]                     ),
      .sldu_alu_valid_i     (sldu_alu_valid_i                       ),
      .sldu_alu_ready_o     (sldu_alu_ready_o_                      ),
      // Interface with the Slide Unit
      .alu_red_ready_i      (sldu_alu_gnt_i                         ),
      // Interface with the Mask unit
      .mask_operand_o       (mask_operand_o_alu                     ),
      .mask_operand_valid_o (mask_operand_valid_o_alu               ),
      .mask_operand_ready_i (mask_operand_ready_i[MaskFUAlu]        ),
      .mask_i               (mask_i_[i]                             ),
      .mask_valid_i         (mask_valid_i                           ),
      .mask_ready_o         (alu_mask_ready_alu                     )
    );

  end

  ///////////////////
  //  Vector MFPU  //
  ///////////////////
 
  // wys
  logic   [Nr_SIMD-1:0]                mfpu_ready_o_;
  logic   [Nr_SIMD-1:0][NrVInsn-1:0]   mfpu_vinsn_done_o_;
  logic   [Nr_SIMD-1:0][2:0]           mfpu_operand_ready_o_;
  logic   [Nr_SIMD-1:0]                mfpu_result_req_o_;
  vid_t   [Nr_SIMD-1:0]                mfpu_result_id_o_;
  vaddr_t [Nr_SIMD-1:0]                mfpu_result_addr_o_;

  assign mfpu_ready_o         = &mfpu_ready_o_;
  assign mfpu_result_req_o    = |mfpu_result_req_o_;
  assign mfpu_result_id_o     = mfpu_result_id_o_[0];
  assign mfpu_result_addr_o   = mfpu_result_addr_o_[0];

  for(genvar i=0;i<Nr_SIMD;i++) begin :gen_mfpu
    elen_t [2:0] mfpu_operand_i_;
    elen_t       mfpu_result_wdata_o_;
    logic  [7:0] mfpu_result_be_o_;

    assign mfpu_operand_ready_o                     = (i==0)?mfpu_operand_ready_o_[i]:mfpu_operand_ready_o_[i] & mfpu_operand_ready_o;
    assign mfpu_vinsn_done_o                        = (i==0)?mfpu_vinsn_done_o_[i]:mfpu_vinsn_done_o_[i] & mfpu_vinsn_done_o;
    assign mfpu_operand_i_[0]                       = mfpu_operand_i[0][ELEN*(i+1)-1:ELEN*i];
    assign mfpu_operand_i_[1]                       = mfpu_operand_i[1][ELEN*(i+1)-1:ELEN*i];
    assign mfpu_operand_i_[2]                       = mfpu_operand_i[2][ELEN*(i+1)-1:ELEN*i];
    assign mfpu_result_wdata_o[ELEN*(i+1)-1:ELEN*i] = mfpu_result_wdata_o_;
    assign mfpu_result_be_o[8*(i+1)-1:8*i]          = mfpu_result_be_o_;

    logic sldu_mfpu_req_valid_o_;
    logic sldu_mfpu_ready_o_;

    assign sldu_mfpu_req_valid_o[i] = sldu_mfpu_req_valid_o_;
    assign sldu_mfpu_ready_o        = (i==0)?sldu_mfpu_ready_o_:sldu_mfpu_ready_o_ & sldu_mfpu_ready_o;

    elen_t mask_operand_o_mfpu;
    logic  mask_operand_valid_o_mfpu;
    logic  mfpu_mask_ready_mfpu;

    assign mask_operand_o[MaskFUMFpu][ELEN*(i+1)-1:ELEN*i] = mask_operand_o_mfpu;
    assign mask_operand_valid_o[MaskFUMFpu]                = (i==0)?mask_operand_valid_o_mfpu:mask_operand_valid_o_mfpu & mask_operand_valid_o[MaskFUMFpu];
    assign mfpu_mask_ready                                 = (i==0)?mfpu_mask_ready_mfpu:mfpu_mask_ready_mfpu & mfpu_mask_ready;

    vmfpu #(
    .NrLanes(NrLanes),
    .FPUSupport(FPUSupport),
    .FPExtSupport(FPExtSupport),
    .FixPtSupport(FixPtSupport),
    .vaddr_t(vaddr_t)
    ) i_vmfpu (
      .clk_i                (clk_i                                    ),
      .rst_ni               (rst_ni                                   ),
      .lane_id_i            (i                                        ),
      // Interface with Dispatcher  
      .mfpu_vxsat_o         (mfpu_vxsat                               ),
      .mfpu_vxrm_i          (alu_vxrm_i                               ),
      // Interface with CVA6
      .fflags_ex_o          (fflags_ex_o                              ),
      .fflags_ex_valid_o    (fflags_ex_valid_o                        ),
      // Interface with the lane sequencer
      .vfu_operation_i      (vfu_operation_i                          ),
      .vfu_operation_valid_i(vfu_operation_valid_i                    ),
      .mfpu_ready_o         (mfpu_ready_o_[i]                         ),
      .mfpu_vinsn_done_o    (mfpu_vinsn_done_o_[i]                    ),
      // Interface with the operand queues
      // wys
      .mfpu_operand_i       (mfpu_operand_i_                          ),
      .mfpu_operand_valid_i (mfpu_operand_valid_i                     ),
      .mfpu_operand_ready_o (mfpu_operand_ready_o_[i]                 ),
      // Interface with the vector register file
      .mfpu_result_req_o    (mfpu_result_req_o_[i]                    ),
      .mfpu_result_id_o     (mfpu_result_id_o_[i]                     ),
      .mfpu_result_addr_o   (mfpu_result_addr_o_[i]                   ),
      // wys 
      .mfpu_result_wdata_o  (mfpu_result_wdata_o_                     ),
      .mfpu_result_be_o     (mfpu_result_be_o_                        ),
      .mfpu_result_gnt_i    (mfpu_result_gnt_i & mfpu_result_req_o_[i]),
      // Interface
      .mfpu_red_valid_o     (sldu_mfpu_req_valid_o_                   ),
      .sldu_operand_i       (sldu_operand_i_[i]                       ),
      .sldu_mfpu_valid_i    (sldu_mfpu_valid_i                        ),
      .sldu_mfpu_ready_o    (sldu_mfpu_ready_o_                       ),
      .mfpu_red_ready_i     (sldu_mfpu_gnt_i                          ),
      // Interface with the Mask unit
      .mask_operand_o       (mask_operand_o_mfpu                      ),
      .mask_operand_valid_o (mask_operand_valid_o_mfpu                ),
      .mask_operand_ready_i (mask_operand_ready_i[MaskFUMFpu]         ),
      .mask_i               (mask_i_[i]                               ),
      .mask_valid_i         (mask_valid_i                             ),
      .mask_ready_o         (mfpu_mask_ready_mfpu                     )
      );
  end

endmodule : vector_fus_stage
