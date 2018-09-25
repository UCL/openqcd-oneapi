# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 17.0.4.196 Build 20170411";
# mark_description "-I../../..//include -I.. -I/cineca/prod/opt/compilers/intel/pe-xe-2017/binary/impi/2017.3.196/intel64/includ";
# mark_description "e -isystem /cineca/prod/opt/compilers/intel/pe-xe-2018/binary/impi/2018.1.163/include64/ -DNPROC0=1 -DNPROC1";
# mark_description "=1 -DNPROC2=1 -DNPROC3=1 -DL0=8 -DL1=8 -DL2=8 -DL3=8 -DNPROC0_BLK=1 -DNPROC1_BLK=1 -DNPROC2_BLK=1 -DNPROC3_B";
# mark_description "LK=1 -std=c89 -xCORE-AVX512 -mtune=skylake -DAVX512 -O3 -Ddirac_counters -pedantic -fstrict-aliasing -Wno-lo";
# mark_description "ng-long -Wstrict-prototypes -S";
	.file "Dw_dble_avx512.c"
	.text
..TXTST0:
# -- Begin  doe_dble_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl doe_dble_avx512
# --- doe_dble_avx512(const int *, const int *, const su3_dble *, const spinor_dble *, double, double, double, spin_t *)
doe_dble_avx512:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %xmm0
# parameter 6: %xmm1
# parameter 7: %xmm2
# parameter 8: %r8
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_doe_dble_avx512.1:
..L2:
                                                          #21.1
        movslq    (%rdi), %rax                                  #37.16
        movslq    (%rsi), %r9                                   #38.16
        vmulsd    %xmm0, %xmm2, %xmm30                          #24.40
        vmovups   .L_2il0floatpacket.14(%rip), %zmm13           #40.3
        vmovups   .L_2il0floatpacket.15(%rip), %zmm17           #40.3
        vmovups   .L_2il0floatpacket.16(%rip), %zmm14           #40.3
        vmovups   .L_2il0floatpacket.17(%rip), %zmm18           #40.3
        vmovsd    %xmm30, -8(%rsp)                              #24.22
        vmovups   .L_2il0floatpacket.18(%rip), %zmm30           #40.3
        vmovaps   %zmm13, %zmm27                                #40.3
        lea       (%rax,%rax,2), %r11                           #37.8
        shlq      $6, %r11                                      #37.8
        lea       (%r9,%r9,2), %r10                             #38.8
        shlq      $6, %r10                                      #38.8
        movl      $15, %eax                                     #43.8
        vmovaps   %zmm13, %zmm26                                #41.3
        kmovw     %eax, %k4                                     #43.8
        movl      $240, %eax                                    #44.8
        kmovw     %eax, %k3                                     #44.8
        vmovups   (%rcx,%r10), %zmm29                           #40.3
        vmovups   (%rcx,%r11), %zmm23                           #40.3
        vmovups   96(%rcx,%r10), %zmm28                         #41.3
        vmovups   96(%rcx,%r11), %zmm22                         #41.3
        vpermi2pd %zmm23, %zmm29, %zmm27                        #40.3
        vpermt2pd 64(%rcx,%r11), %zmm14, %zmm23                 #40.3
        vpermt2pd 64(%rcx,%r10), %zmm17, %zmm29                 #40.3
        vpermi2pd %zmm22, %zmm28, %zmm26                        #41.3
        vpermt2pd 160(%rcx,%r10), %zmm17, %zmm28                #41.3
        vpermt2pd 160(%rcx,%r11), %zmm14, %zmm22                #41.3
        vaddpd    %zmm26, %zmm27, %zmm3{%k4}{z}                 #43.8
        movslq    4(%rdi), %rax                                 #50.16
        vmovaps   %zmm18, %zmm25                                #40.3
        vmovaps   %zmm18, %zmm24                                #41.3
        vpermi2pd %zmm29, %zmm23, %zmm25                        #40.3
        lea       (%rax,%rax,2), %rax                           #50.8
        vpermt2pd %zmm29, %zmm30, %zmm23                        #40.3
        vpermi2pd %zmm28, %zmm22, %zmm24                        #41.3
        vpermt2pd %zmm28, %zmm30, %zmm22                        #41.3
        vsubpd    %zmm26, %zmm27, %zmm3{%k3}                    #44.8
        vaddpd    %zmm24, %zmm25, %zmm7{%k4}{z}                 #45.8
        vaddpd    %zmm22, %zmm23, %zmm8{%k4}{z}                 #47.8
        vsubpd    %zmm24, %zmm25, %zmm7{%k3}                    #46.8
        vsubpd    %zmm22, %zmm23, %zmm8{%k3}                    #48.8
        shlq      $6, %rax                                      #50.8
        prefetcht0 (%rcx,%rax)                                  #51.3
        movslq    4(%rsi), %r9                                  #52.16
        vpermilpd $85, %zmm3, %zmm21                            #57.3
        vpermilpd $85, %zmm7, %zmm20                            #57.3
        vpermilpd $85, %zmm8, %zmm28                            #57.3
        lea       (%r9,%r9,2), %r10                             #52.8
        movl      $90, %r9d                                     #71.8
        kmovw     %r9d, %k1                                     #71.8
        movl      $165, %r9d                                    #72.8
        kmovw     %r9d, %k2                                     #72.8
        vbroadcastsd %xmm1, %zmm16                              #60.13
        shlq      $6, %r10                                      #52.8
        movl      $175, %r9d                                    #89.3
        kmovw     %r9d, %k5                                     #89.3
        movl      $80, %r9d                                     #89.3
        kmovw     %r9d, %k6                                     #89.3
        movl      $60, %r9d                                     #99.8
        kmovw     %r9d, %k7                                     #99.8
        prefetcht0 (%rcx,%r10)                                  #53.3
        vmovups   .L_2il0floatpacket.19(%rip), %zmm29           #57.3
        vmovups   (%rdx), %zmm9                                 #57.3
        vmovups   .L_2il0floatpacket.25(%rip), %zmm23           #57.3
        vmovups   64(%rdx), %zmm15                              #57.3
        vmovups   128(%rdx), %zmm4                              #57.3
        vmovups   96(%rcx,%r10), %zmm1                          #68.3
        vmulpd    %zmm20, %zmm29, %zmm11                        #57.3
        vmulpd    %zmm28, %zmm29, %zmm20                        #57.3
        vmulpd    %zmm29, %zmm21, %zmm6                         #57.3
        vmovups   .L_2il0floatpacket.20(%rip), %zmm28           #57.3
        vmovups   .L_2il0floatpacket.27(%rip), %zmm21           #57.3
        vmovaps   %zmm9, %zmm27                                 #57.3
        vpermt2pd 144(%rdx), %zmm28, %zmm27                     #57.3
        vmulpd    %zmm3, %zmm27, %zmm26                         #57.3
        vmovups   .L_2il0floatpacket.21(%rip), %zmm27           #57.3
        vmovaps   %zmm9, %zmm25                                 #57.3
        vpermt2pd 144(%rdx), %zmm27, %zmm25                     #57.3
        vfmadd213pd %zmm26, %zmm6, %zmm25                       #57.3
        vmovups   .L_2il0floatpacket.22(%rip), %zmm26           #57.3
        vmovaps   %zmm9, %zmm24                                 #57.3
        vpermt2pd 144(%rdx), %zmm26, %zmm24                     #57.3
        vfmadd213pd %zmm25, %zmm7, %zmm24                       #57.3
        vmovups   .L_2il0floatpacket.23(%rip), %zmm25           #57.3
        vmovaps   %zmm9, %zmm10                                 #57.3
        vpermt2pd 144(%rdx), %zmm25, %zmm10                     #57.3
        vfmadd213pd %zmm24, %zmm11, %zmm10                      #57.3
        vmovups   .L_2il0floatpacket.24(%rip), %zmm24           #57.3
        vmovaps   %zmm9, %zmm22                                 #57.3
        vpermt2pd 208(%rdx), %zmm24, %zmm22                     #57.3
        vfmadd213pd %zmm10, %zmm8, %zmm22                       #57.3
        vmovaps   %zmm9, %zmm10                                 #57.3
        vpermt2pd 208(%rdx), %zmm23, %zmm10                     #57.3
        vfmadd213pd %zmm22, %zmm20, %zmm10                      #57.3
        vmovups   .L_2il0floatpacket.26(%rip), %zmm22           #57.3
        vmovaps   %zmm9, %zmm12                                 #57.3
        vpermt2pd 144(%rdx), %zmm22, %zmm12                     #57.3
        vpermt2pd 144(%rdx), %zmm21, %zmm9                      #57.3
        vmulpd    %zmm12, %zmm3, %zmm5                          #57.3
        vfmadd213pd %zmm5, %zmm6, %zmm9                         #57.3
        vmovaps   %zmm15, %zmm5                                 #57.3
        vpermt2pd 144(%rdx), %zmm24, %zmm5                      #57.3
        vmulpd    %zmm5, %zmm3, %zmm3                           #57.3
        vmovaps   %zmm15, %zmm2                                 #57.3
        vpermt2pd 144(%rdx), %zmm23, %zmm2                      #57.3
        vfmadd213pd %zmm3, %zmm6, %zmm2                         #57.3
        vmovaps   %zmm15, %zmm19                                #57.3
        vmovaps   %zmm15, %zmm6                                 #57.3
        vpermt2pd 208(%rdx), %zmm28, %zmm19                     #57.3
        vpermt2pd 208(%rdx), %zmm22, %zmm6                      #57.3
        vfmadd213pd %zmm9, %zmm7, %zmm19                        #57.3
        vfmadd213pd %zmm2, %zmm7, %zmm6                         #57.3
        vmovups   (%rcx,%r10), %zmm7                            #67.3
        vmovaps   %zmm15, %zmm31                                #57.3
        vpermt2pd 208(%rdx), %zmm27, %zmm31                     #57.3
        vmovaps   %zmm15, %zmm12                                #57.3
        vmovaps   %zmm15, %zmm9                                 #57.3
        vpermt2pd 208(%rdx), %zmm21, %zmm15                     #57.3
        vpermt2pd 208(%rdx), %zmm26, %zmm12                     #57.3
        vpermt2pd 208(%rdx), %zmm25, %zmm9                      #57.3
        vfmadd213pd %zmm19, %zmm11, %zmm31                      #57.3
        vfmadd213pd %zmm6, %zmm11, %zmm15                       #57.3
        vmovups   .L_2il0floatpacket.31(%rip), %zmm19           #68.3
        vmulpd    %zmm16, %zmm10, %zmm6                         #61.3
        vfmadd213pd %zmm31, %zmm8, %zmm12                       #57.3
        vmovups   96(%rcx,%rax), %zmm31                         #68.3
        vfmadd213pd %zmm12, %zmm20, %zmm9                       #57.3
        vmovaps   %zmm4, %zmm11                                 #57.3
        vpermt2pd 272(%rdx), %zmm28, %zmm11                     #57.3
        vpermt2pd 272(%rdx), %zmm27, %zmm4                      #57.3
        vfmadd213pd %zmm15, %zmm8, %zmm11                       #57.3
        vmulpd    %zmm9, %zmm16, %zmm8                          #62.3
        vfmadd213pd %zmm11, %zmm20, %zmm4                       #57.3
        vmovups   .L_2il0floatpacket.28(%rip), %zmm20           #61.3
        vmulpd    %zmm4, %zmm16, %zmm4                          #63.3
        vpermpd   %zmm6, %zmm20, %zmm11                         #61.3
        vpermpd   %zmm8, %zmm20, %zmm15                         #62.3
        vpermpd   %zmm4, %zmm20, %zmm16                         #63.3
        vaddpd    %zmm6, %zmm11, %zmm10{%k4}{z}                 #61.3
        vaddpd    %zmm8, %zmm15, %zmm9{%k4}{z}                  #62.3
        vsubpd    %zmm6, %zmm11, %zmm10{%k3}                    #61.3
        vaddpd    %zmm4, %zmm16, %zmm11{%k4}{z}                 #63.3
        vsubpd    %zmm8, %zmm15, %zmm9{%k3}                     #62.3
        vmovups   .L_2il0floatpacket.29(%rip), %zmm15           #68.3
        vsubpd    %zmm4, %zmm16, %zmm11{%k3}                    #63.3
        vmovups   .L_2il0floatpacket.30(%rip), %zmm16           #68.3
        vmovups   (%rcx,%rax), %zmm4                            #67.3
        vmovaps   %zmm1, %zmm0                                  #68.3
        vmovaps   %zmm13, %zmm8                                 #67.3
        vpermt2pd %zmm31, %zmm15, %zmm0                         #68.3
        vpermt2pd 160(%rcx,%r10), %zmm16, %zmm1                 #68.3
        vpermt2pd 160(%rcx,%rax), %zmm19, %zmm31                #68.3
        vpermi2pd %zmm4, %zmm7, %zmm8                           #67.3
        vpermt2pd 64(%rcx,%r10), %zmm17, %zmm7                  #67.3
        vpermt2pd 64(%rcx,%rax), %zmm14, %zmm4                  #67.3
        vmovaps   %zmm18, %zmm3                                 #68.3
        vmovaps   %zmm18, %zmm6                                 #67.3
        vpermi2pd %zmm1, %zmm31, %zmm3                          #68.3
        vpermi2pd %zmm7, %zmm4, %zmm6                           #67.3
        vpermt2pd %zmm7, %zmm30, %zmm4                          #67.3
        vpermt2pd %zmm1, %zmm30, %zmm31                         #68.3
        vpermilpd $85, %zmm3, %zmm12                            #73.8
        movslq    8(%rdi), %r11                                 #80.16
        vpermilpd $85, %zmm0, %zmm7                             #70.8
        vaddpd    %zmm12, %zmm6, %zmm5{%k1}{z}                  #74.8
        vaddpd    %zmm7, %zmm8, %zmm2{%k1}{z}                   #71.8
        vsubpd    %zmm12, %zmm6, %zmm5{%k2}                     #75.8
        vsubpd    %zmm7, %zmm8, %zmm2{%k2}                      #72.8
        vpermilpd $85, %zmm31, %zmm6                            #76.8
        lea       (%r11,%r11,2), %r10                           #80.8
        shlq      $6, %r10                                      #80.8
        vaddpd    %zmm6, %zmm4, %zmm8{%k1}{z}                   #77.8
        vsubpd    %zmm6, %zmm4, %zmm8{%k2}                      #78.8
        prefetcht0 (%rcx,%r10)                                  #81.3
        movslq    8(%rsi), %rax                                 #82.16
        vpermilpd $85, %zmm2, %zmm4                             #87.3
        vpermilpd $85, %zmm5, %zmm1                             #87.3
        vpermilpd $85, %zmm8, %zmm0                             #87.3
        lea       (%rax,%rax,2), %r9                            #82.8
        shlq      $6, %r9                                       #82.8
        movl      $63, %eax                                     #114.3
        kmovw     %eax, %k1                                     #114.3
        movl      $192, %eax                                    #114.3
        kmovw     %eax, %k2                                     #114.3
        vmulpd    %zmm29, %zmm4, %zmm3                          #87.3
        vmulpd    %zmm1, %zmm29, %zmm4                          #87.3
        vmulpd    %zmm0, %zmm29, %zmm7                          #87.3
        prefetcht0 (%rcx,%r9)                                   #83.3
        movl      $195, %eax                                    #98.8
        vmovups   288(%rdx), %zmm1                              #87.3
        vmovups   352(%rdx), %zmm12                             #87.3
        vmovups   416(%rdx), %zmm6                              #87.3
        vmovaps   %zmm1, %zmm31                                 #87.3
        vpermt2pd 432(%rdx), %zmm28, %zmm31                     #87.3
        vmulpd    %zmm2, %zmm31, %zmm0                          #87.3
        vmovaps   %zmm1, %zmm31                                 #87.3
        vpermt2pd 432(%rdx), %zmm27, %zmm31                     #87.3
        vfmadd213pd %zmm0, %zmm3, %zmm31                        #87.3
        vmovaps   %zmm1, %zmm0                                  #87.3
        vpermt2pd 432(%rdx), %zmm26, %zmm0                      #87.3
        vfmadd213pd %zmm31, %zmm5, %zmm0                        #87.3
        vmovaps   %zmm1, %zmm31                                 #87.3
        vpermt2pd 432(%rdx), %zmm25, %zmm31                     #87.3
        vfmadd213pd %zmm0, %zmm4, %zmm31                        #87.3
        vmovaps   %zmm1, %zmm0                                  #87.3
        vpermt2pd 496(%rdx), %zmm24, %zmm0                      #87.3
        vfmadd213pd %zmm31, %zmm8, %zmm0                        #87.3
        vmovaps   %zmm1, %zmm31                                 #87.3
        vpermt2pd 496(%rdx), %zmm23, %zmm31                     #87.3
        vfmadd213pd %zmm0, %zmm7, %zmm31                        #87.3
        vmovaps   %zmm1, %zmm0                                  #87.3
        vpermt2pd 432(%rdx), %zmm22, %zmm0                      #87.3
        vpermt2pd 432(%rdx), %zmm21, %zmm1                      #87.3
        vmulpd    %zmm0, %zmm2, %zmm0                           #87.3
        vfmadd213pd %zmm0, %zmm3, %zmm1                         #87.3
        vmovaps   %zmm12, %zmm0                                 #87.3
        vpermt2pd 496(%rdx), %zmm28, %zmm0                      #87.3
        vfmadd213pd %zmm1, %zmm5, %zmm0                         #87.3
        vmovaps   %zmm12, %zmm1                                 #87.3
        vpermt2pd 496(%rdx), %zmm27, %zmm1                      #87.3
        vfmadd213pd %zmm0, %zmm4, %zmm1                         #87.3
        vmovaps   %zmm12, %zmm0                                 #87.3
        vpermt2pd 496(%rdx), %zmm26, %zmm0                      #87.3
        vfmadd213pd %zmm1, %zmm8, %zmm0                         #87.3
        vmovaps   %zmm12, %zmm1                                 #87.3
        vpermt2pd 496(%rdx), %zmm25, %zmm1                      #87.3
        vfmadd213pd %zmm0, %zmm7, %zmm1                         #87.3
        vmovaps   %zmm12, %zmm0                                 #87.3
        vpermt2pd 432(%rdx), %zmm24, %zmm0                      #87.3
        vmulpd    %zmm0, %zmm2, %zmm2                           #87.3
        vmovaps   %zmm12, %zmm0                                 #87.3
        vpermt2pd 432(%rdx), %zmm23, %zmm0                      #87.3
        vfmadd213pd %zmm2, %zmm3, %zmm0                         #87.3
        vmovaps   %zmm12, %zmm3                                 #87.3
        vpermt2pd 496(%rdx), %zmm22, %zmm3                      #87.3
        vpermt2pd 496(%rdx), %zmm21, %zmm12                     #87.3
        vfmadd213pd %zmm0, %zmm5, %zmm3                         #87.3
        vmovups   (%rcx,%r9), %zmm0                             #95.3
        vfmadd213pd %zmm3, %zmm4, %zmm12                        #87.3
        vpermpd   %zmm31, %zmm20, %zmm4                         #89.3
        vmovaps   %zmm6, %zmm5                                  #87.3
        vpermt2pd 560(%rdx), %zmm28, %zmm5                      #87.3
        vpermt2pd 560(%rdx), %zmm27, %zmm6                      #87.3
        vaddpd    %zmm31, %zmm4, %zmm4{%k4}                     #89.3
        vfmadd213pd %zmm12, %zmm8, %zmm5                        #87.3
        vmovups   .L_2il0floatpacket.32(%rip), %zmm12           #89.3
        vsubpd    %zmm4, %zmm31, %zmm4{%k3}                     #89.3
        vpermpd   %zmm1, %zmm20, %zmm8                          #90.3
        vfmadd213pd %zmm5, %zmm7, %zmm6                         #87.3
        vpermpd   %zmm4, %zmm12, %zmm7                          #89.3
        vaddpd    %zmm1, %zmm8, %zmm8{%k4}                      #90.3
        vpermpd   %zmm6, %zmm20, %zmm2                          #91.3
        vaddpd    %zmm7, %zmm10, %zmm10{%k5}                    #89.3
        vsubpd    %zmm8, %zmm1, %zmm8{%k3}                      #90.3
        vaddpd    %zmm6, %zmm2, %zmm2{%k4}                      #91.3
        vsubpd    %zmm7, %zmm10, %zmm10{%k6}                    #89.3
        vpermpd   %zmm8, %zmm12, %zmm3                          #90.3
        vsubpd    %zmm2, %zmm6, %zmm2{%k3}                      #91.3
        vmovups   (%rcx,%r10), %zmm7                            #95.3
        vmovups   96(%rcx,%r9), %zmm1                           #96.3
        vmovups   96(%rcx,%r10), %zmm8                          #96.3
        vpermpd   %zmm2, %zmm12, %zmm6                          #91.3
        vaddpd    %zmm3, %zmm9, %zmm9{%k5}                      #90.3
        vpermi2pd %zmm8, %zmm1, %zmm15                          #96.3
        vaddpd    %zmm6, %zmm11, %zmm11{%k5}                    #91.3
        vpermt2pd 160(%rcx,%r10), %zmm19, %zmm8                 #96.3
        vpermt2pd 160(%rcx,%r9), %zmm16, %zmm1                  #96.3
        vsubpd    %zmm6, %zmm11, %zmm11{%k6}                    #91.3
        vsubpd    %zmm3, %zmm9, %zmm9{%k6}                      #90.3
        kmovw     %eax, %k5                                     #98.8
        vmovaps   %zmm13, %zmm31                                #95.3
        vpermi2pd %zmm7, %zmm0, %zmm31                          #95.3
        vpermt2pd 64(%rcx,%r9), %zmm17, %zmm0                   #95.3
        vpermt2pd 64(%rcx,%r10), %zmm14, %zmm7                  #95.3
        vaddpd    %zmm15, %zmm31, %zmm2{%k5}{z}                 #98.8
        movslq    12(%rdi), %rdi                                #105.15
        vmovaps   %zmm18, %zmm6                                 #95.3
        vmovaps   %zmm18, %zmm16                                #96.3
        vpermi2pd %zmm0, %zmm7, %zmm6                           #95.3
        lea       (%rdi,%rdi,2), %r9                            #105.8
        vpermt2pd %zmm0, %zmm30, %zmm7                          #95.3
        vpermi2pd %zmm1, %zmm8, %zmm16                          #96.3
        vpermt2pd %zmm1, %zmm30, %zmm8                          #96.3
        vsubpd    %zmm15, %zmm31, %zmm2{%k7}                    #99.8
        vaddpd    %zmm16, %zmm6, %zmm3{%k5}{z}                  #100.8
        vaddpd    %zmm8, %zmm7, %zmm4{%k5}{z}                   #102.8
        vsubpd    %zmm16, %zmm6, %zmm3{%k7}                     #101.8
        vsubpd    %zmm8, %zmm7, %zmm4{%k7}                      #103.8
        shlq      $6, %r9                                       #105.8
        prefetcht0 (%rcx,%r9)                                   #106.3
        movslq    12(%rsi), %rsi                                #107.15
        vpermilpd $85, %zmm2, %zmm19                            #112.3
        vpermilpd $85, %zmm3, %zmm6                             #112.3
        vpermilpd $85, %zmm4, %zmm15                            #112.3
        lea       (%rsi,%rsi,2), %rax                           #107.8
        shlq      $6, %rax                                      #107.8
        vmulpd    %zmm29, %zmm19, %zmm5                         #112.3
        vmulpd    %zmm6, %zmm29, %zmm12                         #112.3
        vmulpd    %zmm15, %zmm29, %zmm8                         #112.3
        prefetcht0 (%rcx,%rax)                                  #108.3
        vmovups   576(%rdx), %zmm19                             #112.3
        vmovups   640(%rdx), %zmm7                              #112.3
        vmovups   704(%rdx), %zmm6                              #112.3
        vmovaps   %zmm19, %zmm16                                #112.3
        vpermt2pd 720(%rdx), %zmm28, %zmm16                     #112.3
        vmulpd    %zmm2, %zmm16, %zmm15                         #112.3
        vmovaps   %zmm19, %zmm0                                 #112.3
        vpermt2pd 720(%rdx), %zmm27, %zmm0                      #112.3
        vfmadd213pd %zmm15, %zmm5, %zmm0                        #112.3
        vmovaps   %zmm19, %zmm1                                 #112.3
        vpermt2pd 720(%rdx), %zmm26, %zmm1                      #112.3
        vfmadd213pd %zmm0, %zmm3, %zmm1                         #112.3
        vmovaps   %zmm19, %zmm31                                #112.3
        vmovaps   %zmm19, %zmm0                                 #112.3
        vpermt2pd 720(%rdx), %zmm25, %zmm31                     #112.3
        vpermt2pd 720(%rdx), %zmm22, %zmm0                      #112.3
        vfmadd213pd %zmm1, %zmm12, %zmm31                       #112.3
        vmulpd    %zmm0, %zmm2, %zmm1                           #112.3
        vmovaps   %zmm19, %zmm15                                #112.3
        vpermt2pd 784(%rdx), %zmm24, %zmm15                     #112.3
        vfmadd213pd %zmm31, %zmm4, %zmm15                       #112.3
        vmovaps   %zmm19, %zmm16                                #112.3
        vpermt2pd 720(%rdx), %zmm21, %zmm19                     #112.3
        vpermt2pd 784(%rdx), %zmm23, %zmm16                     #112.3
        vfmadd213pd %zmm1, %zmm5, %zmm19                        #112.3
        vfmadd213pd %zmm15, %zmm8, %zmm16                       #112.3
        vmovaps   %zmm7, %zmm15                                 #112.3
        vpermt2pd 784(%rdx), %zmm28, %zmm15                     #112.3
        vfmadd213pd %zmm19, %zmm3, %zmm15                       #112.3
        vmovaps   %zmm7, %zmm19                                 #112.3
        vpermt2pd 784(%rdx), %zmm27, %zmm19                     #112.3
        vmovaps   %zmm7, %zmm1                                  #112.3
        vpermt2pd 720(%rdx), %zmm24, %zmm1                      #112.3
        vfmadd213pd %zmm15, %zmm12, %zmm19                      #112.3
        vmulpd    %zmm1, %zmm2, %zmm2                           #112.3
        vmovaps   %zmm7, %zmm0                                  #112.3
        vpermt2pd 784(%rdx), %zmm26, %zmm0                      #112.3
        vfmadd213pd %zmm19, %zmm4, %zmm0                        #112.3
        vmovaps   %zmm7, %zmm15                                 #112.3
        vpermt2pd 784(%rdx), %zmm25, %zmm15                     #112.3
        vfmadd213pd %zmm0, %zmm8, %zmm15                        #112.3
        vmovaps   %zmm7, %zmm0                                  #112.3
        vpermt2pd 720(%rdx), %zmm23, %zmm0                      #112.3
        vfmadd213pd %zmm2, %zmm5, %zmm0                         #112.3
        vmovaps   %zmm7, %zmm5                                  #112.3
        vpermt2pd 784(%rdx), %zmm22, %zmm5                      #112.3
        vpermt2pd 784(%rdx), %zmm21, %zmm7                      #112.3
        vfmadd213pd %zmm0, %zmm3, %zmm5                         #112.3
        vmovups   .L_2il0floatpacket.33(%rip), %zmm0            #114.3
        vfmadd213pd %zmm5, %zmm12, %zmm7                        #112.3
        vmovaps   %zmm6, %zmm3                                  #112.3
        vpermt2pd 848(%rdx), %zmm28, %zmm3                      #112.3
        vpermt2pd 848(%rdx), %zmm27, %zmm6                      #112.3
        vfmadd213pd %zmm7, %zmm4, %zmm3                         #112.3
        vpermpd   %zmm16, %zmm20, %zmm4                         #114.3
        vfmadd213pd %zmm3, %zmm8, %zmm6                         #112.3
        vmovups   96(%rcx,%r9), %zmm3                           #121.3
        vpermpd   %zmm15, %zmm20, %zmm8                         #115.3
        vpermpd   %zmm6, %zmm20, %zmm1                          #116.3
        vaddpd    %zmm16, %zmm4, %zmm4{%k4}                     #114.3
        vaddpd    %zmm15, %zmm8, %zmm8{%k4}                     #115.3
        vaddpd    %zmm6, %zmm1, %zmm1{%k4}                      #116.3
        vsubpd    %zmm4, %zmm16, %zmm4{%k3}                     #114.3
        vsubpd    %zmm8, %zmm15, %zmm8{%k3}                     #115.3
        vsubpd    %zmm1, %zmm6, %zmm1{%k3}                      #116.3
        vpermpd   %zmm4, %zmm0, %zmm7                           #114.3
        vpermpd   %zmm8, %zmm0, %zmm12                          #115.3
        vpermpd   %zmm1, %zmm0, %zmm2                           #116.3
        vaddpd    %zmm7, %zmm10, %zmm10{%k1}                    #114.3
        vaddpd    %zmm12, %zmm9, %zmm9{%k1}                     #115.3
        vaddpd    %zmm2, %zmm11, %zmm11{%k1}                    #116.3
        vsubpd    %zmm7, %zmm10, %zmm10{%k2}                    #114.3
        vsubpd    %zmm12, %zmm9, %zmm9{%k2}                     #115.3
        vsubpd    %zmm2, %zmm11, %zmm11{%k2}                    #116.3
        vmovups   (%rcx,%rax), %zmm6                            #120.3
        vmovups   (%rcx,%r9), %zmm4                             #120.3
        vmovups   96(%rcx,%rax), %zmm0                          #121.3
        vmovaps   %zmm13, %zmm1                                 #120.3
        vpermi2pd %zmm4, %zmm6, %zmm1                           #120.3
        vpermt2pd 64(%rcx,%rax), %zmm17, %zmm6                  #120.3
        vpermt2pd 64(%rcx,%r9), %zmm14, %zmm4                   #120.3
        vpermi2pd %zmm3, %zmm0, %zmm13                          #121.3
        vpermt2pd 160(%rcx,%rax), %zmm17, %zmm0                 #121.3
        vpermt2pd 160(%rcx,%r9), %zmm14, %zmm3                  #121.3
        vmovaps   %zmm18, %zmm2                                 #120.3
        vpermi2pd %zmm6, %zmm4, %zmm2                           #120.3
        vpermt2pd %zmm6, %zmm30, %zmm4                          #120.3
        vpermi2pd %zmm0, %zmm3, %zmm18                          #121.3
                                # LOE rdx rbx rbp r8 r12 r13 r14 r15 zmm0 zmm1 zmm2 zmm3 zmm4 zmm9 zmm10 zmm11 zmm13 zmm18 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 k3 k4
..B1.4:                         # Preds ..B1.1
                                # Execution count [1.00e+00]
        movl      $150, %eax                                    #124.8
        kmovw     %eax, %k1                                     #124.8
        movl      $105, %eax                                    #125.8
        kmovw     %eax, %k2                                     #125.8
        vpermt2pd %zmm0, %zmm30, %zmm3                          #121.3
        vmovups   928(%rdx), %zmm5                              #135.3
        vmovups   992(%rdx), %zmm7                              #135.3
        vmovups   1136(%rdx), %zmm6                             #135.3
        vpermilpd $85, %zmm18, %zmm15                           #126.8
        movl      $111, %eax                                    #137.3
        vaddpd    %zmm15, %zmm2, %zmm12{%k1}{z}                 #127.8
        kmovw     %eax, %k5                                     #137.3
        vsubpd    %zmm15, %zmm2, %zmm12{%k2}                    #128.8
        vmovups   1008(%rdx), %zmm2                             #135.3
        vpermilpd $85, %zmm13, %zmm14                           #123.8
        movl      $144, %eax                                    #137.3
        vpermilpd $85, %zmm3, %zmm16                            #129.8
        vmovups   864(%rdx), %zmm3                              #135.3
        vaddpd    %zmm14, %zmm1, %zmm13{%k1}{z}                 #124.8
        vaddpd    %zmm16, %zmm4, %zmm8{%k1}{z}                  #130.8
        kmovw     %eax, %k6                                     #137.3
        vsubpd    %zmm14, %zmm1, %zmm13{%k2}                    #125.8
        vsubpd    %zmm16, %zmm4, %zmm8{%k2}                     #131.8
        vmovups   1072(%rdx), %zmm4                             #135.3
        vmovaps   %zmm28, %zmm30                                #135.3
        vpermi2pd %zmm2, %zmm3, %zmm30                          #135.3
        vpermilpd $85, %zmm13, %zmm17                           #135.3
        vmulpd    %zmm29, %zmm17, %zmm1                         #135.3
        vmulpd    %zmm13, %zmm30, %zmm31                        #135.3
        vpermilpd $85, %zmm12, %zmm18                           #135.3
        vmulpd    %zmm18, %zmm29, %zmm0                         #135.3
        vmovaps   %zmm22, %zmm18                                #135.3
        vmovaps   %zmm24, %zmm17                                #135.3
        vpermi2pd %zmm2, %zmm3, %zmm18                          #135.3
        vpermi2pd %zmm2, %zmm5, %zmm24                          #135.3
        vpermi2pd %zmm4, %zmm3, %zmm17                          #135.3
        vpermi2pd %zmm4, %zmm5, %zmm22                          #135.3
        vmulpd    %zmm24, %zmm13, %zmm24                        #135.3
        vmovaps   %zmm27, %zmm14                                #135.3
        vpermilpd $85, %zmm8, %zmm19                            #135.3
        vpermi2pd %zmm2, %zmm3, %zmm14                          #135.3
        vmulpd    %zmm19, %zmm29, %zmm29                        #135.3
        vmulpd    %zmm18, %zmm13, %zmm19                        #135.3
        vfmadd213pd %zmm31, %zmm1, %zmm14                       #135.3
        vmovaps   %zmm26, %zmm15                                #135.3
        vpermi2pd %zmm2, %zmm3, %zmm15                          #135.3
        vpermi2pd %zmm4, %zmm5, %zmm26                          #135.3
        vfmadd213pd %zmm14, %zmm12, %zmm15                      #135.3
        vmovaps   %zmm25, %zmm16                                #135.3
        vmovaps   %zmm23, %zmm14                                #135.3
        vpermi2pd %zmm2, %zmm3, %zmm16                          #135.3
        vpermi2pd %zmm4, %zmm3, %zmm14                          #135.3
        vpermt2pd %zmm2, %zmm21, %zmm3                          #135.3
        vpermi2pd %zmm2, %zmm5, %zmm23                          #135.3
        vpermi2pd %zmm4, %zmm5, %zmm25                          #135.3
        vfmadd213pd %zmm19, %zmm1, %zmm3                        #135.3
        vfmadd213pd %zmm24, %zmm1, %zmm23                       #135.3
        vfmadd213pd %zmm15, %zmm0, %zmm16                       #135.3
        vfmadd213pd %zmm23, %zmm12, %zmm22                      #135.3
        vfmadd213pd %zmm16, %zmm8, %zmm17                       #135.3
        vmovaps   %zmm28, %zmm30                                #135.3
        vpermi2pd %zmm4, %zmm5, %zmm30                          #135.3
        vpermi2pd %zmm6, %zmm7, %zmm28                          #135.3
        vpermt2pd %zmm6, %zmm27, %zmm7                          #135.3
        vfmadd213pd %zmm3, %zmm12, %zmm30                       #135.3
        vfmadd213pd %zmm17, %zmm29, %zmm14                      #135.3
        vmovaps   %zmm27, %zmm3                                 #135.3
        vpermi2pd %zmm4, %zmm5, %zmm3                           #135.3
        vpermt2pd %zmm4, %zmm21, %zmm5                          #135.3
        vpermpd   %zmm14, %zmm20, %zmm21                        #137.3
        vfmadd213pd %zmm30, %zmm0, %zmm3                        #135.3
        vfmadd213pd %zmm22, %zmm0, %zmm5                        #135.3
        vmovups   .L_2il0floatpacket.36(%rip), %zmm0            #147.3
        vaddpd    %zmm14, %zmm21, %zmm21{%k4}                   #137.3
        vfmadd213pd %zmm3, %zmm8, %zmm26                        #135.3
        vfmadd213pd %zmm5, %zmm8, %zmm28                        #135.3
        vsubpd    %zmm21, %zmm14, %zmm21{%k3}                   #137.3
        vfmadd213pd %zmm26, %zmm29, %zmm25                      #135.3
        vfmadd213pd %zmm28, %zmm29, %zmm7                       #135.3
        vmovups   .L_2il0floatpacket.34(%rip), %zmm26           #137.3
        vpermpd   %zmm25, %zmm20, %zmm23                        #138.3
        vpermpd   %zmm7, %zmm20, %zmm20                         #139.3
        vpermpd   %zmm21, %zmm26, %zmm22                        #137.3
        vaddpd    %zmm25, %zmm23, %zmm23{%k4}                   #138.3
        vaddpd    %zmm7, %zmm20, %zmm20{%k4}                    #139.3
        vaddpd    %zmm22, %zmm10, %zmm10{%k5}                   #137.3
        vsubpd    %zmm23, %zmm25, %zmm23{%k3}                   #138.3
        vsubpd    %zmm20, %zmm7, %zmm20{%k3}                    #139.3
        vsubpd    %zmm22, %zmm10, %zmm10{%k6}                   #137.3
        vpermpd   %zmm23, %zmm26, %zmm25                        #138.3
        vpermpd   %zmm20, %zmm26, %zmm27                        #139.3
        vbroadcastsd -8(%rsp), %zmm28                           #142.13
        vaddpd    %zmm25, %zmm9, %zmm9{%k5}                     #138.3
        vaddpd    %zmm27, %zmm11, %zmm11{%k5}                   #139.3
        vmulpd    %zmm10, %zmm28, %zmm10                        #143.8
        vsubpd    %zmm25, %zmm9, %zmm9{%k6}                     #138.3
        vsubpd    %zmm27, %zmm11, %zmm11{%k6}                   #139.3
        vmulpd    %zmm9, %zmm28, %zmm1                          #144.8
        vmulpd    %zmm11, %zmm28, %zmm9                         #145.8
        vmovups   .L_2il0floatpacket.35(%rip), %zmm28           #147.3
        vmovups   .L_2il0floatpacket.37(%rip), %zmm11           #147.3
        vpermi2pd %zmm10, %zmm9, %zmm0                          #147.3
        vpermi2pd %zmm1, %zmm10, %zmm28                         #147.3
        vpermt2pd %zmm9, %zmm11, %zmm1                          #147.3
        vmovupd   %ymm28, (%r8)                                 #147.3
        vmovupd   %ymm0, 32(%r8)                                #147.3
        vmovupd   %ymm1, 64(%r8)                                #147.3
        vextractf64x4 $1, %zmm28, 96(%r8)                       #147.3
        vextractf64x4 $1, %zmm0, 128(%r8)                       #147.3
        vextractf64x4 $1, %zmm1, 160(%r8)                       #147.3
        vzeroupper                                              #148.1
        ret                                                     #148.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	doe_dble_avx512,@function
	.size	doe_dble_avx512,.-doe_dble_avx512
	.data
# -- End  doe_dble_avx512
	.text
# -- Begin  deo_dble_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl deo_dble_avx512
# --- deo_dble_avx512(const int *, const int *, const su3_dble *, spinor_dble *, double, double, double, spin_t *)
deo_dble_avx512:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %xmm0
# parameter 6: %xmm1
# parameter 7: %xmm2
# parameter 8: %r8
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_deo_dble_avx512.4:
..L5:
                                                          #151.1
        pushq     %rbp                                          #151.1
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp                                    #151.1
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp                                    #151.1
        movslq    (%rdi), %rax                                  #165.16
        lea       (%rax,%rax,2), %r11                           #165.8
        shlq      $6, %r11                                      #165.8
        prefetcht0 (%rcx,%r11)                                  #166.3
        movl      $15, %eax                                     #178.3
        movslq    (%rsi), %r9                                   #167.16
        kmovw     %eax, %k5                                     #178.3
        movl      $240, %eax                                    #178.3
        kmovw     %eax, %k6                                     #178.3
        vbroadcastsd %xmm0, %zmm24                              #173.11
        vbroadcastsd %xmm2, %zmm13                              #196.11
        movl      $90, %eax                                     #201.3
        lea       (%r9,%r9,2), %r10                             #167.8
        shlq      $6, %r10                                      #167.8
        kmovw     %eax, %k1                                     #201.3
        movl      $165, %eax                                    #201.3
        kmovw     %eax, %k2                                     #201.3
        movl      $195, %eax                                    #218.3
        kmovw     %eax, %k4                                     #218.3
        movl      $60, %eax                                     #218.3
        kmovw     %eax, %k3                                     #218.3
        prefetcht0 (%rcx,%r10)                                  #168.3
        vmovups   96(%r8), %zmm27                               #170.3
        vmovups   (%r8), %zmm23                                 #170.3
        vmovups   .L_2il0floatpacket.14(%rip), %zmm26           #170.3
        vmovups   .L_2il0floatpacket.15(%rip), %zmm30           #170.3
        vmovups   .L_2il0floatpacket.16(%rip), %zmm29           #170.3
        vmovups   .L_2il0floatpacket.17(%rip), %zmm25           #170.3
        vmovups   .L_2il0floatpacket.18(%rip), %zmm28           #170.3
        vmovups   .L_2il0floatpacket.28(%rip), %zmm18           #178.3
        vmovups   144(%rdx), %zmm21                             #184.3
        vmovups   208(%rdx), %zmm11                             #184.3
        vmovups   272(%rdx), %zmm8                              #184.3
        vmovups   96(%rcx,%r10), %zmm2                          #187.3
        vpermi2pd %zmm23, %zmm27, %zmm26                        #170.3
        vpermt2pd 160(%r8), %zmm30, %zmm27                      #170.3
        vpermt2pd 64(%r8), %zmm29, %zmm23                       #170.3
        vmulpd    %zmm26, %zmm24, %zmm6                         #174.8
        vpermi2pd %zmm27, %zmm23, %zmm25                        #170.3
        vpermt2pd %zmm27, %zmm28, %zmm23                        #170.3
        vpermpd   %zmm6, %zmm18, %zmm22                         #178.3
        vmulpd    %zmm25, %zmm24, %zmm1                         #175.8
        vmulpd    %zmm23, %zmm24, %zmm14                        #176.8
        vaddpd    %zmm6, %zmm22, %zmm10{%k5}{z}                 #178.3
        vpermpd   %zmm1, %zmm18, %zmm30                         #179.3
        vpermpd   %zmm14, %zmm18, %zmm29                        #180.3
        vsubpd    %zmm6, %zmm22, %zmm10{%k6}                    #178.3
        vaddpd    %zmm1, %zmm30, %zmm16{%k5}{z}                 #179.3
        vaddpd    %zmm14, %zmm29, %zmm15{%k5}{z}                #180.3
        vmovups   .L_2il0floatpacket.27(%rip), %zmm22           #184.3
        vsubpd    %zmm14, %zmm29, %zmm15{%k6}                   #180.3
        vsubpd    %zmm1, %zmm30, %zmm16{%k6}                    #179.3
        vmovups   .L_2il0floatpacket.20(%rip), %zmm29           #184.3
        vmovups   .L_2il0floatpacket.19(%rip), %zmm30           #184.3
        vmovaps   %zmm21, %zmm25                                #184.3
        vpermt2pd (%rdx), %zmm29, %zmm25                        #184.3
        vpermilpd $85, %zmm10, %zmm28                           #184.3
        vmulpd    %zmm30, %zmm28, %zmm20                        #184.3
        vmulpd    %zmm10, %zmm25, %zmm17                        #184.3
        vmovups   .L_2il0floatpacket.21(%rip), %zmm28           #184.3
        vmovups   .L_2il0floatpacket.24(%rip), %zmm25           #184.3
        vpermilpd $85, %zmm16, %zmm27                           #184.3
        vmovaps   %zmm21, %zmm24                                #184.3
        vmulpd    %zmm27, %zmm30, %zmm19                        #184.3
        vmovups   .L_2il0floatpacket.22(%rip), %zmm27           #184.3
        vpermt2pd (%rdx), %zmm28, %zmm24                        #184.3
        vfmadd213pd %zmm17, %zmm20, %zmm24                      #184.3
        vpermilpd $85, %zmm15, %zmm26                           #184.3
        vmovaps   %zmm21, %zmm23                                #184.3
        vpermt2pd (%rdx), %zmm27, %zmm23                        #184.3
        vmulpd    %zmm26, %zmm30, %zmm18                        #184.3
        vmovups   .L_2il0floatpacket.23(%rip), %zmm26           #184.3
        vfmadd213pd %zmm24, %zmm16, %zmm23                      #184.3
        vmovups   .L_2il0floatpacket.25(%rip), %zmm24           #184.3
        vmovaps   %zmm21, %zmm12                                #184.3
        vpermt2pd (%rdx), %zmm26, %zmm12                        #184.3
        vfmadd213pd %zmm23, %zmm19, %zmm12                      #184.3
        vmovups   .L_2il0floatpacket.26(%rip), %zmm23           #184.3
        vmovaps   %zmm21, %zmm7                                 #184.3
        vpermt2pd (%rdx), %zmm23, %zmm7                         #184.3
        vmulpd    %zmm7, %zmm10, %zmm31                         #184.3
        vmovups   64(%rcx,%r10), %zmm7                          #186.3
        vmovaps   %zmm21, %zmm9                                 #184.3
        vmovaps   %zmm21, %zmm17                                #184.3
        vpermt2pd (%rdx), %zmm22, %zmm21                        #184.3
        vpermt2pd 64(%rdx), %zmm25, %zmm9                       #184.3
        vpermt2pd 64(%rdx), %zmm24, %zmm17                      #184.3
        vfmadd213pd %zmm31, %zmm20, %zmm21                      #184.3
        vfmadd213pd %zmm12, %zmm15, %zmm9                       #184.3
        vmovaps   %zmm11, %zmm5                                 #184.3
        vpermt2pd 64(%rdx), %zmm29, %zmm5                       #184.3
        vfmadd213pd %zmm9, %zmm18, %zmm17                       #184.3
        vmovups   (%rcx,%r10), %zmm9                            #186.3
        vfmadd213pd %zmm21, %zmm16, %zmm5                       #184.3
        vmovaps   %zmm11, %zmm21                                #184.3
        vpermt2pd 64(%rdx), %zmm28, %zmm21                      #184.3
        vfmadd213pd %zmm5, %zmm19, %zmm21                       #184.3
        vmovups   160(%rcx,%r10), %zmm5                         #187.3
        vmovaps   %zmm11, %zmm4                                 #184.3
        vmovaps   %zmm11, %zmm3                                 #184.3
        vpermt2pd 64(%rdx), %zmm27, %zmm4                       #184.3
        vpermt2pd (%rdx), %zmm25, %zmm3                         #184.3
        vfmadd213pd %zmm21, %zmm15, %zmm4                       #184.3
        vmulpd    %zmm3, %zmm10, %zmm21                         #184.3
        vmovaps   %zmm11, %zmm10                                #184.3
        vpermt2pd (%rdx), %zmm24, %zmm10                        #184.3
        vfmadd213pd %zmm21, %zmm20, %zmm10                      #184.3
        vmovups   .L_2il0floatpacket.38(%rip), %zmm21           #186.3
        vmovaps   %zmm11, %zmm20                                #184.3
        vpermt2pd 64(%rdx), %zmm23, %zmm20                      #184.3
        vpermt2pd 160(%rcx,%r11), %zmm21, %zmm5                 #187.3
        vpermt2pd 64(%rcx,%r11), %zmm21, %zmm7                  #186.3
        vfmadd213pd %zmm10, %zmm16, %zmm20                      #184.3
        vmovaps   %zmm11, %zmm12                                #184.3
        vpermt2pd 64(%rdx), %zmm22, %zmm11                      #184.3
        vpermt2pd 64(%rdx), %zmm26, %zmm12                      #184.3
        vfmadd213pd %zmm20, %zmm19, %zmm11                      #184.3
        vfmadd213pd %zmm4, %zmm18, %zmm12                       #184.3
        vmovups   .L_2il0floatpacket.39(%rip), %zmm20           #186.3
        vmovaps   %zmm8, %zmm19                                 #184.3
        vpermt2pd 128(%rdx), %zmm29, %zmm19                     #184.3
        vpermt2pd 128(%rdx), %zmm28, %zmm8                      #184.3
        vfmadd213pd %zmm11, %zmm15, %zmm19                      #184.3
        vfmadd213pd %zmm19, %zmm18, %zmm8                       #184.3
        vmovups   .L_2il0floatpacket.36(%rip), %zmm18           #186.3
        vmovups   .L_2il0floatpacket.35(%rip), %zmm19           #186.3
        vmovaps   %zmm8, %zmm15                                 #186.3
        vmovaps   %zmm17, %zmm16                                #186.3
        vpermt2pd %zmm17, %zmm18, %zmm15                        #186.3
        vpermt2pd %zmm12, %zmm19, %zmm16                        #186.3
        vmovups   .L_2il0floatpacket.37(%rip), %zmm17           #186.3
        vmovaps   %zmm2, %zmm31                                 #187.3
        vpermt2pd %zmm8, %zmm17, %zmm12                         #186.3
        vpermt2pd 96(%rcx,%r11), %zmm21, %zmm31                 #187.3
        vpermt2pd 96(%rcx,%r11), %zmm20, %zmm2                  #187.3
        vaddpd    %zmm12, %zmm5, %zmm5{%k5}                     #187.3
        vaddpd    %zmm16, %zmm31, %zmm31{%k5}                   #187.3
        vaddpd    %zmm15, %zmm2, %zmm2{%k5}                     #187.3
        vaddpd    %zmm12, %zmm7, %zmm8                          #186.3
        vsubpd    %zmm16, %zmm31, %zmm31{%k6}                   #187.3
        vsubpd    %zmm15, %zmm2, %zmm2{%k6}                     #187.3
        vsubpd    %zmm12, %zmm5, %zmm5{%k6}                     #187.3
        vmovaps   %zmm9, %zmm11                                 #186.3
        vpermt2pd (%rcx,%r11), %zmm21, %zmm11                   #186.3
        vpermt2pd (%rcx,%r11), %zmm20, %zmm9                    #186.3
        vaddpd    %zmm16, %zmm11, %zmm11                        #186.3
        vaddpd    %zmm15, %zmm9, %zmm10                         #186.3
        movslq    4(%rdi), %rax                                 #190.16
        vmovupd   %ymm11, (%rcx,%r10)                           #186.3
        vmovupd   %ymm10, 32(%rcx,%r10)                         #186.3
        vmovupd   %ymm8, 64(%rcx,%r10)                          #186.3
        vextractf64x4 $1, %zmm11, (%rcx,%r11)                   #186.3
        vextractf64x4 $1, %zmm10, 32(%rcx,%r11)                 #186.3
        vextractf64x4 $1, %zmm8, 64(%rcx,%r11)                  #186.3
        vmovupd   %ymm31, 96(%rcx,%r10)                         #187.3
        vmovupd   %ymm2, 128(%rcx,%r10)                         #187.3
        vmovupd   %ymm5, 160(%rcx,%r10)                         #187.3
        vextractf64x4 $1, %zmm31, 96(%rcx,%r11)                 #187.3
        vextractf64x4 $1, %zmm2, 128(%rcx,%r11)                 #187.3
        vextractf64x4 $1, %zmm5, 160(%rcx,%r11)                 #187.3
        lea       (%rax,%rax,2), %r10                           #190.8
        shlq      $6, %r10                                      #190.8
        prefetcht0 (%rcx,%r10)                                  #191.3
        movslq    4(%rsi), %r8                                  #192.16
        vmulpd    %zmm6, %zmm13, %zmm16                         #197.8
        vmulpd    %zmm1, %zmm13, %zmm15                         #198.8
        vmulpd    %zmm14, %zmm13, %zmm14                        #199.8
        lea       (%r8,%r8,2), %r9                              #192.8
        shlq      $6, %r9                                       #192.8
        prefetcht0 (%rcx,%r9)                                   #193.3
        vmovups   .L_2il0floatpacket.40(%rip), %zmm1            #201.3
        vmovups   .L_2il0floatpacket.41(%rip), %zmm3            #201.3
        vmovups   496(%rdx), %zmm7                              #207.3
        vmovups   560(%rdx), %zmm31                             #207.3
        vpermpd   %zmm16, %zmm1, %zmm12                         #201.3
        vpermpd   %zmm16, %zmm3, %zmm13                         #201.3
        vpermpd   %zmm15, %zmm1, %zmm6                          #202.3
        vpermpd   %zmm14, %zmm1, %zmm0                          #203.3
        vaddpd    %zmm12, %zmm13, %zmm4{%k1}{z}                 #201.3
        vpermpd   %zmm14, %zmm3, %zmm11                         #203.3
        vsubpd    %zmm12, %zmm13, %zmm4{%k2}                    #201.3
        vpermpd   %zmm15, %zmm3, %zmm12                         #202.3
        vaddpd    %zmm0, %zmm11, %zmm2{%k1}{z}                  #203.3
        vaddpd    %zmm6, %zmm12, %zmm5{%k1}{z}                  #202.3
        vsubpd    %zmm0, %zmm11, %zmm2{%k2}                     #203.3
        vsubpd    %zmm6, %zmm12, %zmm5{%k2}                     #202.3
        vmovups   432(%rdx), %zmm6                              #207.3
        vpermilpd $85, %zmm4, %zmm10                            #207.3
        vmulpd    %zmm30, %zmm10, %zmm9                         #207.3
        vmovaps   %zmm6, %zmm10                                 #207.3
        vpermt2pd 288(%rdx), %zmm29, %zmm10                     #207.3
        vmulpd    %zmm4, %zmm10, %zmm0                          #207.3
        vmovaps   %zmm6, %zmm10                                 #207.3
        vpermt2pd 288(%rdx), %zmm28, %zmm10                     #207.3
        vpermilpd $85, %zmm5, %zmm8                             #207.3
        vmulpd    %zmm8, %zmm30, %zmm1                          #207.3
        vfmadd213pd %zmm0, %zmm9, %zmm10                        #207.3
        vpermilpd $85, %zmm2, %zmm3                             #207.3
        vmulpd    %zmm3, %zmm30, %zmm8                          #207.3
        vmovaps   %zmm6, %zmm3                                  #207.3
        vpermt2pd 288(%rdx), %zmm27, %zmm3                      #207.3
        vfmadd213pd %zmm10, %zmm5, %zmm3                        #207.3
        vmovaps   %zmm6, %zmm10                                 #207.3
        vpermt2pd 288(%rdx), %zmm26, %zmm10                     #207.3
        vfmadd213pd %zmm3, %zmm1, %zmm10                        #207.3
        vmovaps   %zmm6, %zmm0                                  #207.3
        vpermt2pd 352(%rdx), %zmm25, %zmm0                      #207.3
        vfmadd213pd %zmm10, %zmm2, %zmm0                        #207.3
        vmovaps   %zmm6, %zmm10                                 #207.3
        vmovaps   %zmm6, %zmm3                                  #207.3
        vpermt2pd 352(%rdx), %zmm24, %zmm10                     #207.3
        vpermt2pd 288(%rdx), %zmm23, %zmm3                      #207.3
        vpermt2pd 288(%rdx), %zmm22, %zmm6                      #207.3
        vfmadd213pd %zmm0, %zmm8, %zmm10                        #207.3
        vmulpd    %zmm3, %zmm4, %zmm0                           #207.3
        vfmadd213pd %zmm0, %zmm9, %zmm6                         #207.3
        vmovaps   %zmm7, %zmm3                                  #207.3
        vpermt2pd 352(%rdx), %zmm29, %zmm3                      #207.3
        vfmadd213pd %zmm6, %zmm5, %zmm3                         #207.3
        vmovaps   %zmm7, %zmm6                                  #207.3
        vpermt2pd 352(%rdx), %zmm28, %zmm6                      #207.3
        vfmadd213pd %zmm3, %zmm1, %zmm6                         #207.3
        vmovaps   %zmm7, %zmm3                                  #207.3
        vpermt2pd 288(%rdx), %zmm25, %zmm3                      #207.3
        vmulpd    %zmm3, %zmm4, %zmm4                           #207.3
        vmovaps   %zmm7, %zmm3                                  #207.3
        vpermt2pd 288(%rdx), %zmm24, %zmm3                      #207.3
        vfmadd213pd %zmm4, %zmm9, %zmm3                         #207.3
        vmovaps   %zmm7, %zmm9                                  #207.3
        vpermt2pd 352(%rdx), %zmm23, %zmm9                      #207.3
        vmovaps   %zmm7, %zmm0                                  #207.3
        vpermt2pd 352(%rdx), %zmm27, %zmm0                      #207.3
        vfmadd213pd %zmm3, %zmm5, %zmm9                         #207.3
        vfmadd213pd %zmm6, %zmm2, %zmm0                         #207.3
        vmovaps   %zmm7, %zmm6                                  #207.3
        vpermt2pd 352(%rdx), %zmm22, %zmm7                      #207.3
        vpermt2pd 352(%rdx), %zmm26, %zmm6                      #207.3
        vfmadd213pd %zmm9, %zmm1, %zmm7                         #207.3
        vfmadd213pd %zmm0, %zmm8, %zmm6                         #207.3
        vmovups   64(%rcx,%r9), %zmm0                           #209.3
        vmovups   .L_2il0floatpacket.42(%rip), %zmm9            #210.3
        vpermt2pd 64(%rcx,%r10), %zmm21, %zmm0                  #209.3
        vmovups   %zmm9, -64(%rsp)                              #210.3[spill]
        vmovaps   %zmm31, %zmm1                                 #207.3
        vpermt2pd 416(%rdx), %zmm29, %zmm1                      #207.3
        vpermt2pd 416(%rdx), %zmm28, %zmm31                     #207.3
        vfmadd213pd %zmm7, %zmm2, %zmm1                         #207.3
        vmovups   (%rcx,%r9), %zmm2                             #209.3
        vfmadd213pd %zmm1, %zmm8, %zmm31                        #207.3
        vmovaps   %zmm2, %zmm8                                  #209.3
        vmovaps   %zmm10, %zmm7                                 #209.3
        vmovaps   %zmm31, %zmm5                                 #209.3
        vmovaps   %zmm6, %zmm1                                  #209.3
        vpermt2pd (%rcx,%r10), %zmm21, %zmm8                    #209.3
        vpermt2pd (%rcx,%r10), %zmm20, %zmm2                    #209.3
        vpermt2pd %zmm6, %zmm19, %zmm7                          #209.3
        vpermt2pd %zmm10, %zmm18, %zmm5                         #209.3
        vpermt2pd %zmm31, %zmm17, %zmm1                         #209.3
        vaddpd    %zmm7, %zmm8, %zmm4                           #209.3
        vaddpd    %zmm5, %zmm2, %zmm8                           #209.3
        vaddpd    %zmm1, %zmm0, %zmm1                           #209.3
        vmovups   .L_2il0floatpacket.43(%rip), %zmm2            #210.3
        vmovups   160(%rcx,%r9), %zmm5                          #210.3
        vpermi2pd %zmm10, %zmm31, %zmm2                         #210.3
        vpermt2pd 160(%rcx,%r10), %zmm21, %zmm5                 #210.3
        vmovaps   %zmm10, %zmm7                                 #210.3
        vmovups   .L_2il0floatpacket.44(%rip), %zmm10           #210.3
        vpermt2pd %zmm6, %zmm9, %zmm7                           #210.3
        vpermt2pd %zmm31, %zmm10, %zmm6                         #210.3
        movslq    8(%rdi), %r11                                 #213.16
        vaddpd    %zmm6, %zmm5, %zmm5{%k2}                      #210.3
        vsubpd    %zmm6, %zmm5, %zmm5{%k1}                      #210.3
        lea       (%r11,%r11,2), %r8                            #213.8
        shlq      $6, %r8                                       #213.8
        vmovupd   %ymm4, (%rcx,%r9)                             #209.3
        vmovupd   %ymm8, 32(%rcx,%r9)                           #209.3
        vmovupd   %ymm1, 64(%rcx,%r9)                           #209.3
        vextractf64x4 $1, %zmm1, 64(%rcx,%r10)                  #209.3
        vmovups   96(%rcx,%r9), %zmm1                           #210.3
        vextractf64x4 $1, %zmm8, 32(%rcx,%r10)                  #209.3
        vextractf64x4 $1, %zmm4, (%rcx,%r10)                    #209.3
        vmovaps   %zmm1, %zmm8                                  #210.3
        vpermt2pd 96(%rcx,%r10), %zmm21, %zmm8                  #210.3
        vpermt2pd 96(%rcx,%r10), %zmm20, %zmm1                  #210.3
        vaddpd    %zmm7, %zmm8, %zmm8{%k2}                      #210.3
        vaddpd    %zmm2, %zmm1, %zmm1{%k2}                      #210.3
        vsubpd    %zmm7, %zmm8, %zmm8{%k1}                      #210.3
        vsubpd    %zmm2, %zmm1, %zmm1{%k1}                      #210.3
        vmovupd   %ymm8, 96(%rcx,%r9)                           #210.3
        vmovupd   %ymm1, 128(%rcx,%r9)                          #210.3
        vmovupd   %ymm5, 160(%rcx,%r9)                          #210.3
        vextractf64x4 $1, %zmm8, 96(%rcx,%r10)                  #210.3
        vextractf64x4 $1, %zmm1, 128(%rcx,%r10)                 #210.3
        vextractf64x4 $1, %zmm5, 160(%rcx,%r10)                 #210.3
        prefetcht0 (%rcx,%r8)                                   #214.3
        movslq    8(%rsi), %rax                                 #215.16
        lea       (%rax,%rax,2), %rax                           #215.8
        shlq      $6, %rax                                      #215.8
        prefetcht0 (%rcx,%rax)                                  #216.3
        vmovups   .L_2il0floatpacket.45(%rip), %zmm8            #218.3
        vmovups   784(%rdx), %zmm9                              #224.3
        vpermpd   %zmm14, %zmm8, %zmm1                          #220.3
        vpermpd   %zmm15, %zmm8, %zmm31                         #219.3
        vpermpd   %zmm16, %zmm8, %zmm6                          #218.3
        vaddpd    %zmm1, %zmm11, %zmm8{%k4}{z}                  #220.3
        vaddpd    %zmm31, %zmm12, %zmm5{%k4}{z}                 #219.3
        vaddpd    %zmm6, %zmm13, %zmm2{%k4}{z}                  #218.3
        vsubpd    %zmm1, %zmm11, %zmm8{%k3}                     #220.3
        vsubpd    %zmm31, %zmm12, %zmm5{%k3}                    #219.3
        vsubpd    %zmm6, %zmm13, %zmm2{%k3}                     #218.3
        vmovups   720(%rdx), %zmm1                              #224.3
        vmovups   848(%rdx), %zmm6                              #224.3
        vpermilpd $85, %zmm5, %zmm31                            #224.3
        vmulpd    %zmm31, %zmm30, %zmm4                         #224.3
        vmovaps   %zmm1, %zmm31                                 #224.3
        vpermt2pd 576(%rdx), %zmm29, %zmm31                     #224.3
        vpermilpd $85, %zmm2, %zmm7                             #224.3
        vpermilpd $85, %zmm8, %zmm0                             #224.3
        vmulpd    %zmm30, %zmm7, %zmm3                          #224.3
        vmulpd    %zmm0, %zmm30, %zmm7                          #224.3
        vmulpd    %zmm2, %zmm31, %zmm0                          #224.3
        vmovaps   %zmm1, %zmm31                                 #224.3
        vpermt2pd 576(%rdx), %zmm28, %zmm31                     #224.3
        vfmadd213pd %zmm0, %zmm3, %zmm31                        #224.3
        vmovaps   %zmm1, %zmm0                                  #224.3
        vpermt2pd 576(%rdx), %zmm27, %zmm0                      #224.3
        vfmadd213pd %zmm31, %zmm5, %zmm0                        #224.3
        vmovaps   %zmm1, %zmm31                                 #224.3
        vpermt2pd 576(%rdx), %zmm26, %zmm31                     #224.3
        vfmadd213pd %zmm0, %zmm4, %zmm31                        #224.3
        vmovaps   %zmm1, %zmm0                                  #224.3
        vpermt2pd 640(%rdx), %zmm25, %zmm0                      #224.3
        vfmadd213pd %zmm31, %zmm8, %zmm0                        #224.3
        vmovaps   %zmm1, %zmm31                                 #224.3
        vpermt2pd 640(%rdx), %zmm24, %zmm31                     #224.3
        vfmadd213pd %zmm0, %zmm7, %zmm31                        #224.3
        vmovaps   %zmm1, %zmm0                                  #224.3
        vpermt2pd 576(%rdx), %zmm23, %zmm0                      #224.3
        vpermt2pd 576(%rdx), %zmm22, %zmm1                      #224.3
        vmulpd    %zmm0, %zmm2, %zmm0                           #224.3
        vfmadd213pd %zmm0, %zmm3, %zmm1                         #224.3
        vmovaps   %zmm9, %zmm0                                  #224.3
        vpermt2pd 640(%rdx), %zmm29, %zmm0                      #224.3
        vfmadd213pd %zmm1, %zmm5, %zmm0                         #224.3
        vmovaps   %zmm9, %zmm1                                  #224.3
        vpermt2pd 640(%rdx), %zmm28, %zmm1                      #224.3
        vfmadd213pd %zmm0, %zmm4, %zmm1                         #224.3
        vmovaps   %zmm9, %zmm0                                  #224.3
        vpermt2pd 640(%rdx), %zmm27, %zmm0                      #224.3
        vfmadd213pd %zmm1, %zmm8, %zmm0                         #224.3
        vmovaps   %zmm9, %zmm1                                  #224.3
        vpermt2pd 640(%rdx), %zmm26, %zmm1                      #224.3
        vfmadd213pd %zmm0, %zmm7, %zmm1                         #224.3
        vmovaps   %zmm9, %zmm0                                  #224.3
        vpermt2pd 576(%rdx), %zmm25, %zmm0                      #224.3
        vmulpd    %zmm0, %zmm2, %zmm0                           #224.3
        vmovaps   %zmm9, %zmm2                                  #224.3
        vpermt2pd 576(%rdx), %zmm24, %zmm2                      #224.3
        vfmadd213pd %zmm0, %zmm3, %zmm2                         #224.3
        vmovaps   %zmm9, %zmm3                                  #224.3
        vpermt2pd 640(%rdx), %zmm23, %zmm3                      #224.3
        vpermt2pd 640(%rdx), %zmm22, %zmm9                      #224.3
        vfmadd213pd %zmm2, %zmm5, %zmm3                         #224.3
        vfmadd213pd %zmm3, %zmm4, %zmm9                         #224.3
        vmovups   (%rcx,%rax), %zmm3                            #226.3
        vmovaps   %zmm6, %zmm0                                  #224.3
        vpermt2pd 704(%rdx), %zmm29, %zmm0                      #224.3
        vpermt2pd 704(%rdx), %zmm28, %zmm6                      #224.3
        vfmadd213pd %zmm9, %zmm8, %zmm0                         #224.3
        vfmadd213pd %zmm0, %zmm7, %zmm6                         #224.3
        vmovups   64(%rcx,%rax), %zmm7                          #226.3
        vmovaps   %zmm3, %zmm0                                  #226.3
        vmovaps   %zmm31, %zmm2                                 #226.3
        vmovaps   %zmm6, %zmm4                                  #226.3
        vmovaps   %zmm1, %zmm8                                  #226.3
        vpermt2pd (%rcx,%r8), %zmm21, %zmm0                     #226.3
        vpermt2pd (%rcx,%r8), %zmm20, %zmm3                     #226.3
        vpermt2pd 64(%rcx,%r8), %zmm21, %zmm7                   #226.3
        vpermt2pd %zmm1, %zmm19, %zmm2                          #226.3
        vpermt2pd %zmm31, %zmm18, %zmm4                         #226.3
        vpermt2pd %zmm6, %zmm17, %zmm8                          #226.3
                                # LOE rax rdx rcx rbx rsi rdi r8 r12 r13 r14 r15 zmm0 zmm1 zmm2 zmm3 zmm4 zmm6 zmm7 zmm8 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31 k1 k2 k3 k4 k5 k6
..B2.4:                         # Preds ..B2.1
                                # Execution count [1.00e+00]
        vaddpd    %zmm2, %zmm0, %zmm5                           #226.3
        vaddpd    %zmm4, %zmm3, %zmm9                           #226.3
        vaddpd    %zmm8, %zmm7, %zmm7                           #226.3
        vmovups   .L_2il0floatpacket.46(%rip), %zmm8            #227.3
        vmovups   96(%rcx,%r8), %zmm0                           #227.3
        vpermpd   %zmm31, %zmm8, %zmm4                          #227.3
        vpermpd   %zmm1, %zmm8, %zmm1                           #227.3
        vpermpd   %zmm6, %zmm8, %zmm3                           #227.3
        vmovaps   %zmm21, %zmm31                                #227.3
        vmovaps   %zmm19, %zmm2                                 #227.3
        vmovaps   %zmm18, %zmm6                                 #227.3
        vpermi2pd %zmm1, %zmm4, %zmm2                           #227.3
        vpermi2pd %zmm4, %zmm3, %zmm6                           #227.3
        vpermt2pd %zmm3, %zmm17, %zmm1                          #227.3
        vmovupd   %ymm5, (%rcx,%rax)                            #226.3
        vmovupd   %ymm9, 32(%rcx,%rax)                          #226.3
        vmovupd   %ymm7, 64(%rcx,%rax)                          #226.3
        vextractf64x4 $1, %zmm5, (%rcx,%r8)                     #226.3
        vextractf64x4 $1, %zmm7, 64(%rcx,%r8)                   #226.3
        vmovups   96(%rcx,%rax), %zmm5                          #227.3
        vmovups   160(%rcx,%rax), %zmm7                         #227.3
        vextractf64x4 $1, %zmm9, 32(%rcx,%r8)                   #226.3
        vpermi2pd %zmm0, %zmm5, %zmm31                          #227.3
        vpermt2pd %zmm0, %zmm20, %zmm5                          #227.3
        vpermt2pd 160(%rcx,%r8), %zmm21, %zmm7                  #227.3
        vaddpd    %zmm2, %zmm31, %zmm31{%k6}                    #227.3
        vaddpd    %zmm6, %zmm5, %zmm5{%k3}                      #227.3
        vaddpd    %zmm1, %zmm7, %zmm7{%k5}                      #227.3
        vsubpd    %zmm2, %zmm31, %zmm31{%k5}                    #227.3
        vsubpd    %zmm6, %zmm5, %zmm5{%k4}                      #227.3
        vsubpd    %zmm1, %zmm7, %zmm7{%k6}                      #227.3
        vmovupd   %ymm31, 96(%rcx,%rax)                         #227.3
        vmovupd   %ymm5, 128(%rcx,%rax)                         #227.3
        vmovupd   %ymm7, 160(%rcx,%rax)                         #227.3
        vextractf64x4 $1, %zmm31, 96(%rcx,%r8)                  #227.3
        vextractf64x4 $1, %zmm5, 128(%rcx,%r8)                  #227.3
        vextractf64x4 $1, %zmm7, 160(%rcx,%r8)                  #227.3
        movslq    12(%rdi), %rax                                #231.16
        lea       (%rax,%rax,2), %r8                            #231.8
        shlq      $6, %r8                                       #231.8
        prefetcht0 (%rcx,%r8)                                   #232.3
        movl      $150, %eax                                    #236.3
        movslq    12(%rsi), %rsi                                #233.16
        kmovw     %eax, %k4                                     #236.3
        movl      $105, %eax                                    #236.3
        kmovw     %eax, %k3                                     #236.3
        lea       (%rsi,%rsi,2), %rdi                           #233.8
        shlq      $6, %rdi                                      #233.8
        prefetcht0 (%rcx,%rdi)                                  #234.3
        vmovups   .L_2il0floatpacket.47(%rip), %zmm8            #236.3
        vmovups   1072(%rdx), %zmm1                             #242.3
        vmovups   928(%rdx), %zmm6                              #242.3
        vmovups   992(%rdx), %zmm31                             #242.3
        vmovups   1136(%rdx), %zmm0                             #242.3
        vpermpd   %zmm16, %zmm8, %zmm4                          #236.3
        vpermpd   %zmm14, %zmm8, %zmm14                         #238.3
        vaddpd    %zmm4, %zmm13, %zmm16{%k4}{z}                 #236.3
        vaddpd    %zmm14, %zmm11, %zmm2{%k4}{z}                 #238.3
        vsubpd    %zmm4, %zmm13, %zmm16{%k3}                    #236.3
        vpermpd   %zmm15, %zmm8, %zmm13                         #237.3
        vsubpd    %zmm14, %zmm11, %zmm2{%k3}                    #238.3
        vmovups   1008(%rdx), %zmm15                            #242.3
        vaddpd    %zmm13, %zmm12, %zmm3{%k4}{z}                 #237.3
        vsubpd    %zmm13, %zmm12, %zmm3{%k3}                    #237.3
        vmovups   864(%rdx), %zmm13                             #242.3
        vpermilpd $85, %zmm16, %zmm12                           #242.3
        vpermilpd $85, %zmm3, %zmm11                            #242.3
        vpermilpd $85, %zmm2, %zmm9                             #242.3
        vmulpd    %zmm30, %zmm12, %zmm14                        #242.3
        vmulpd    %zmm11, %zmm30, %zmm12                        #242.3
        vmulpd    %zmm9, %zmm30, %zmm11                         #242.3
        vmovaps   %zmm29, %zmm30                                #242.3
        vpermi2pd %zmm13, %zmm15, %zmm30                        #242.3
        vmulpd    %zmm16, %zmm30, %zmm30                        #242.3
        vmovaps   %zmm28, %zmm4                                 #242.3
        vpermi2pd %zmm13, %zmm15, %zmm4                         #242.3
        vfmadd213pd %zmm30, %zmm14, %zmm4                       #242.3
        vmovaps   %zmm27, %zmm5                                 #242.3
        vpermi2pd %zmm13, %zmm15, %zmm5                         #242.3
        vpermi2pd %zmm6, %zmm1, %zmm27                          #242.3
        vfmadd213pd %zmm4, %zmm3, %zmm5                         #242.3
        vmovaps   %zmm23, %zmm4                                 #242.3
        vpermi2pd %zmm13, %zmm15, %zmm4                         #242.3
        vpermi2pd %zmm6, %zmm1, %zmm23                          #242.3
        vmulpd    %zmm4, %zmm16, %zmm9                          #242.3
        vmovaps   %zmm25, %zmm8                                 #242.3
        vpermi2pd %zmm13, %zmm1, %zmm25                         #242.3
        vpermi2pd %zmm6, %zmm15, %zmm8                          #242.3
        vmulpd    %zmm25, %zmm16, %zmm25                        #242.3
        vmovaps   %zmm26, %zmm7                                 #242.3
        vmovaps   %zmm24, %zmm30                                #242.3
        vpermi2pd %zmm13, %zmm15, %zmm7                         #242.3
        vpermi2pd %zmm6, %zmm15, %zmm30                         #242.3
        vpermt2pd %zmm13, %zmm22, %zmm15                        #242.3
        vpermi2pd %zmm13, %zmm1, %zmm24                         #242.3
        vpermi2pd %zmm6, %zmm1, %zmm26                          #242.3
        vfmadd213pd %zmm9, %zmm14, %zmm15                       #242.3
        vfmadd213pd %zmm5, %zmm12, %zmm7                        #242.3
        vfmadd213pd %zmm25, %zmm14, %zmm24                      #242.3
        vfmadd213pd %zmm7, %zmm2, %zmm8                         #242.3
        vfmadd213pd %zmm24, %zmm3, %zmm23                       #242.3
        vmovups   (%rcx,%rdi), %zmm24                           #244.3
        vfmadd213pd %zmm8, %zmm11, %zmm30                       #242.3
        vmovaps   %zmm29, %zmm5                                 #242.3
        vpermi2pd %zmm6, %zmm1, %zmm5                           #242.3
        vpermi2pd %zmm31, %zmm0, %zmm29                         #242.3
        vpermt2pd %zmm31, %zmm28, %zmm0                         #242.3
        vfmadd213pd %zmm15, %zmm3, %zmm5                        #242.3
        vmovaps   %zmm28, %zmm15                                #242.3
        vpermi2pd %zmm6, %zmm1, %zmm15                          #242.3
        vpermt2pd %zmm6, %zmm22, %zmm1                          #242.3
        vmovups   (%rcx,%r8), %zmm22                            #244.3
        vfmadd213pd %zmm5, %zmm12, %zmm15                       #242.3
        vfmadd213pd %zmm23, %zmm12, %zmm1                       #242.3
        vmovups   96(%rcx,%r8), %zmm28                          #245.3
        vfmadd213pd %zmm15, %zmm2, %zmm27                       #242.3
        vfmadd213pd %zmm1, %zmm2, %zmm29                        #242.3
        vmovups   160(%rcx,%rdi), %zmm2                         #245.3
        vmovups   96(%rcx,%rdi), %zmm1                          #245.3
        vfmadd213pd %zmm27, %zmm11, %zmm26                      #242.3
        vmovups   64(%rcx,%rdi), %zmm27                         #244.3
        vfmadd213pd %zmm29, %zmm11, %zmm0                       #242.3
        vpermt2pd 160(%rcx,%r8), %zmm21, %zmm2                  #245.3
        vpermt2pd 64(%rcx,%r8), %zmm21, %zmm27                  #244.3
        vpermi2pd %zmm26, %zmm30, %zmm19                        #244.3
        vpermi2pd %zmm0, %zmm26, %zmm17                         #244.3
        vpermi2pd %zmm26, %zmm30, %zmm10                        #245.3
        vpermi2pd %zmm30, %zmm0, %zmm18                         #244.3
        vaddpd    %zmm17, %zmm27, %zmm17                        #244.3
        vmovaps   %zmm21, %zmm23                                #244.3
        vmovaps   %zmm21, %zmm29                                #245.3
        vpermi2pd %zmm22, %zmm24, %zmm23                        #244.3
        vpermt2pd %zmm22, %zmm20, %zmm24                        #244.3
        vpermi2pd %zmm28, %zmm1, %zmm29                         #245.3
        vpermt2pd %zmm28, %zmm20, %zmm1                         #245.3
        vaddpd    %zmm19, %zmm23, %zmm19                        #244.3
        vaddpd    %zmm10, %zmm29, %zmm29{%k2}                   #245.3
        vaddpd    %zmm18, %zmm24, %zmm18                        #244.3
        vmovups   .L_2il0floatpacket.48(%rip), %zmm21           #245.3
        vmovups   -64(%rsp), %zmm20                             #245.3[spill]
        vsubpd    %zmm10, %zmm29, %zmm29{%k1}                   #245.3
        vpermi2pd %zmm30, %zmm0, %zmm21                         #245.3
        vpermt2pd %zmm0, %zmm20, %zmm26                         #245.3
        vaddpd    %zmm21, %zmm1, %zmm1{%k3}                     #245.3
        vaddpd    %zmm26, %zmm2, %zmm2{%k1}                     #245.3
        vsubpd    %zmm21, %zmm1, %zmm1{%k4}                     #245.3
        vsubpd    %zmm26, %zmm2, %zmm2{%k2}                     #245.3
        vmovupd   %ymm19, (%rcx,%rdi)                           #244.3
        vmovupd   %ymm18, 32(%rcx,%rdi)                         #244.3
        vmovupd   %ymm17, 64(%rcx,%rdi)                         #244.3
        vextractf64x4 $1, %zmm19, (%rcx,%r8)                    #244.3
        vextractf64x4 $1, %zmm18, 32(%rcx,%r8)                  #244.3
        vextractf64x4 $1, %zmm17, 64(%rcx,%r8)                  #244.3
        vmovupd   %ymm29, 96(%rcx,%rdi)                         #245.3
        vmovupd   %ymm1, 128(%rcx,%rdi)                         #245.3
        vmovupd   %ymm2, 160(%rcx,%rdi)                         #245.3
        vextractf64x4 $1, %zmm29, 96(%rcx,%r8)                  #245.3
        vextractf64x4 $1, %zmm1, 128(%rcx,%r8)                  #245.3
        vextractf64x4 $1, %zmm2, 160(%rcx,%r8)                  #245.3
        vzeroupper                                              #246.1
        movq      %rbp, %rsp                                    #246.1
        popq      %rbp                                          #246.1
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret                                                     #246.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	deo_dble_avx512,@function
	.size	deo_dble_avx512,.-deo_dble_avx512
	.data
# -- End  deo_dble_avx512
	.section .rodata, "a"
	.align 64
	.align 64
.L_2il0floatpacket.14:
	.long	0x00000008,0x00000000,0x00000009,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000
	.type	.L_2il0floatpacket.14,@object
	.size	.L_2il0floatpacket.14,64
	.align 64
.L_2il0floatpacket.15:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000
	.type	.L_2il0floatpacket.15,@object
	.size	.L_2il0floatpacket.15,64
	.align 64
.L_2il0floatpacket.16:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.16,@object
	.size	.L_2il0floatpacket.16,64
	.align 64
.L_2il0floatpacket.17:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.17,@object
	.size	.L_2il0floatpacket.17,64
	.align 64
.L_2il0floatpacket.18:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.18,@object
	.size	.L_2il0floatpacket.18,64
	.align 64
.L_2il0floatpacket.19:
	.long	0x00000000,0xbff00000,0x00000000,0x3ff00000,0x00000000,0xbff00000,0x00000000,0x3ff00000,0x00000000,0x3ff00000,0x00000000,0xbff00000,0x00000000,0x3ff00000,0x00000000,0xbff00000
	.type	.L_2il0floatpacket.19,@object
	.size	.L_2il0floatpacket.19,64
	.align 64
.L_2il0floatpacket.20:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000
	.type	.L_2il0floatpacket.20,@object
	.size	.L_2il0floatpacket.20,64
	.align 64
.L_2il0floatpacket.21:
	.long	0x00000001,0x00000000,0x00000001,0x00000000,0x00000001,0x00000000,0x00000001,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000
	.type	.L_2il0floatpacket.21,@object
	.size	.L_2il0floatpacket.21,64
	.align 64
.L_2il0floatpacket.22:
	.long	0x00000002,0x00000000,0x00000002,0x00000000,0x00000002,0x00000000,0x00000002,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.22,@object
	.size	.L_2il0floatpacket.22,64
	.align 64
.L_2il0floatpacket.23:
	.long	0x00000003,0x00000000,0x00000003,0x00000000,0x00000003,0x00000000,0x00000003,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.23,@object
	.size	.L_2il0floatpacket.23,64
	.align 64
.L_2il0floatpacket.24:
	.long	0x00000004,0x00000000,0x00000004,0x00000000,0x00000004,0x00000000,0x00000004,0x00000000,0x0000000c,0x00000000,0x0000000c,0x00000000,0x0000000c,0x00000000,0x0000000c,0x00000000
	.type	.L_2il0floatpacket.24,@object
	.size	.L_2il0floatpacket.24,64
	.align 64
.L_2il0floatpacket.25:
	.long	0x00000005,0x00000000,0x00000005,0x00000000,0x00000005,0x00000000,0x00000005,0x00000000,0x0000000d,0x00000000,0x0000000d,0x00000000,0x0000000d,0x00000000,0x0000000d,0x00000000
	.type	.L_2il0floatpacket.25,@object
	.size	.L_2il0floatpacket.25,64
	.align 64
.L_2il0floatpacket.26:
	.long	0x00000006,0x00000000,0x00000006,0x00000000,0x00000006,0x00000000,0x00000006,0x00000000,0x0000000a,0x00000000,0x0000000a,0x00000000,0x0000000a,0x00000000,0x0000000a,0x00000000
	.type	.L_2il0floatpacket.26,@object
	.size	.L_2il0floatpacket.26,64
	.align 64
.L_2il0floatpacket.27:
	.long	0x00000007,0x00000000,0x00000007,0x00000000,0x00000007,0x00000000,0x00000007,0x00000000,0x0000000b,0x00000000,0x0000000b,0x00000000,0x0000000b,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.27,@object
	.size	.L_2il0floatpacket.27,64
	.align 64
.L_2il0floatpacket.28:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000
	.type	.L_2il0floatpacket.28,@object
	.size	.L_2il0floatpacket.28,64
	.align 64
.L_2il0floatpacket.29:
	.long	0x0000000e,0x00000000,0x0000000f,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000
	.type	.L_2il0floatpacket.29,@object
	.size	.L_2il0floatpacket.29,64
	.align 64
.L_2il0floatpacket.30:
	.long	0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000
	.type	.L_2il0floatpacket.30,@object
	.size	.L_2il0floatpacket.30,64
	.align 64
.L_2il0floatpacket.31:
	.long	0x00000008,0x00000000,0x00000009,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000
	.type	.L_2il0floatpacket.31,@object
	.size	.L_2il0floatpacket.31,64
	.align 64
.L_2il0floatpacket.32:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000
	.type	.L_2il0floatpacket.32,@object
	.size	.L_2il0floatpacket.32,64
	.align 64
.L_2il0floatpacket.33:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000
	.type	.L_2il0floatpacket.33,@object
	.size	.L_2il0floatpacket.33,64
	.align 64
.L_2il0floatpacket.34:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000
	.type	.L_2il0floatpacket.34,@object
	.size	.L_2il0floatpacket.34,64
	.align 64
.L_2il0floatpacket.35:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000
	.type	.L_2il0floatpacket.35,@object
	.size	.L_2il0floatpacket.35,64
	.align 64
.L_2il0floatpacket.36:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.36,@object
	.size	.L_2il0floatpacket.36,64
	.align 64
.L_2il0floatpacket.37:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.37,@object
	.size	.L_2il0floatpacket.37,64
	.align 64
.L_2il0floatpacket.38:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.38,@object
	.size	.L_2il0floatpacket.38,64
	.align 64
.L_2il0floatpacket.39:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.39,@object
	.size	.L_2il0floatpacket.39,64
	.align 64
.L_2il0floatpacket.40:
	.long	0x00000007,0x00000000,0x00000006,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000
	.type	.L_2il0floatpacket.40,@object
	.size	.L_2il0floatpacket.40,64
	.align 64
.L_2il0floatpacket.41:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000
	.type	.L_2il0floatpacket.41,@object
	.size	.L_2il0floatpacket.41,64
	.align 64
.L_2il0floatpacket.42:
	.long	0x00000003,0x00000000,0x00000002,0x00000000,0x0000000b,0x00000000,0x0000000a,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000,0x0000000f,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.42,@object
	.size	.L_2il0floatpacket.42,64
	.align 64
.L_2il0floatpacket.43:
	.long	0x00000003,0x00000000,0x00000002,0x00000000,0x00000009,0x00000000,0x00000008,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000,0x0000000d,0x00000000,0x0000000c,0x00000000
	.type	.L_2il0floatpacket.43,@object
	.size	.L_2il0floatpacket.43,64
	.align 64
.L_2il0floatpacket.44:
	.long	0x00000001,0x00000000,0x00000000,0x00000000,0x00000009,0x00000000,0x00000008,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000,0x0000000d,0x00000000,0x0000000c,0x00000000
	.type	.L_2il0floatpacket.44,@object
	.size	.L_2il0floatpacket.44,64
	.align 64
.L_2il0floatpacket.45:
	.long	0x00000006,0x00000000,0x00000007,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000
	.type	.L_2il0floatpacket.45,@object
	.size	.L_2il0floatpacket.45,64
	.align 64
.L_2il0floatpacket.46:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000
	.type	.L_2il0floatpacket.46,@object
	.size	.L_2il0floatpacket.46,64
	.align 64
.L_2il0floatpacket.47:
	.long	0x00000005,0x00000000,0x00000004,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000,0x00000007,0x00000000,0x00000006,0x00000000
	.type	.L_2il0floatpacket.47,@object
	.size	.L_2il0floatpacket.47,64
	.align 64
.L_2il0floatpacket.48:
	.long	0x00000001,0x00000000,0x00000000,0x00000000,0x0000000b,0x00000000,0x0000000a,0x00000000,0x00000005,0x00000000,0x00000004,0x00000000,0x0000000f,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.48,@object
	.size	.L_2il0floatpacket.48,64
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
