# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 17.0.4.196 Build 20170411";
# mark_description "-I../../..//include -I.. -I/cineca/prod/opt/compilers/intel/pe-xe-2017/binary/impi/2017.3.196/intel64/includ";
# mark_description "e -isystem /cineca/prod/opt/compilers/intel/pe-xe-2018/binary/impi/2018.1.163/include64/ -DNPROC0=1 -DNPROC1";
# mark_description "=1 -DNPROC2=1 -DNPROC3=1 -DL0=8 -DL1=8 -DL2=8 -DL3=8 -DNPROC0_BLK=1 -DNPROC1_BLK=1 -DNPROC2_BLK=1 -DNPROC3_B";
# mark_description "LK=1 -std=c89 -xCORE-AVX512 -mtune=skylake -DAVX512 -O3 -Ddirac_counters -pedantic -fstrict-aliasing -Wno-lo";
# mark_description "ng-long -Wstrict-prototypes -S";
	.file "Dw_avx512.c"
	.text
..TXTST0:
# -- Begin  doe_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl doe_avx512
# --- doe_avx512(int *, int *, su3 *, spinor *)
doe_avx512:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_doe_avx512.1:
..L2:
                                                          #25.1
        pushq     %rbx                                          #25.1
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbx                                    #25.1
	.cfi_def_cfa 3, 16
	.cfi_offset 3, -16
        andq      $-64, %rsp                                    #25.1
        pushq     %rbp                                          #25.1
        pushq     %rbp                                          #25.1
        movq      8(%rbx), %rbp                                 #25.1
        movq      %rbp, 8(%rsp)                                 #25.1
        movq      %rsp, %rbp                                    #25.1
	.cfi_escape 0x10, 0x06, 0x02, 0x76, 0x00
        movslq    (%rdi), %r11                                  #39.16
        movslq    (%rsi), %r8                                   #40.16
        movslq    4(%rdi), %rax                                 #41.17
        vmovups   .L_2il0floatpacket.7(%rip), %zmm28            #47.3
        lea       (%r11,%r11,2), %r10                           #39.8
        vmovups   .L_2il0floatpacket.8(%rip), %zmm11            #47.3
        shlq      $5, %r10                                      #39.8
        lea       (%r8,%r8,2), %r9                              #40.8
        shlq      $5, %r9                                       #40.8
        lea       (%rax,%rax,2), %r8                            #41.9
        movslq    4(%rsi), %r11                                 #42.17
        shlq      $5, %r8                                       #41.9
        vmovups   (%rcx,%r10), %xmm29                           #44.3
        vmovups   16(%rcx,%r10), %xmm10                         #44.3
        vmovups   32(%rcx,%r10), %xmm8                          #44.3
        vmovups   48(%rcx,%r10), %xmm14                         #47.3
        vmovups   64(%rcx,%r10), %xmm12                         #47.3
        vmovups   80(%rcx,%r10), %xmm26                         #47.3
        lea       (%r11,%r11,2), %rax                           #42.9
        shlq      $5, %rax                                      #42.9
        vinsertf32x4 $1, (%rcx,%r9), %zmm29, %zmm21             #44.3
        vinsertf32x4 $1, 16(%rcx,%r9), %zmm10, %zmm16           #44.3
        vinsertf32x4 $2, (%rcx,%r8), %zmm21, %zmm22             #44.3
        vinsertf32x4 $2, 16(%rcx,%r8), %zmm16, %zmm17           #44.3
        vinsertf32x4 $3, (%rcx,%rax), %zmm22, %zmm19            #44.3
        vinsertf32x4 $3, 16(%rcx,%rax), %zmm17, %zmm20          #44.3
        vmovaps   %zmm28, %zmm21                                #47.3
        vshufps   $228, %zmm20, %zmm19, %zmm27                  #44.3
        vinsertf32x4 $1, 32(%rcx,%r9), %zmm8, %zmm9             #44.3
        vinsertf32x4 $1, 48(%rcx,%r9), %zmm14, %zmm0            #47.3
        vinsertf32x4 $1, 64(%rcx,%r9), %zmm12, %zmm7            #47.3
        vinsertf32x4 $1, 80(%rcx,%r9), %zmm26, %zmm29           #47.3
        vinsertf32x4 $2, 32(%rcx,%r8), %zmm9, %zmm18            #44.3
        vinsertf32x4 $2, 48(%rcx,%r8), %zmm0, %zmm15            #47.3
        vinsertf32x4 $2, 64(%rcx,%r8), %zmm7, %zmm24            #47.3
        vinsertf32x4 $2, 80(%rcx,%r8), %zmm29, %zmm5            #47.3
        vinsertf32x4 $3, 32(%rcx,%rax), %zmm18, %zmm13          #44.3
        vinsertf32x4 $3, 48(%rcx,%rax), %zmm15, %zmm10          #47.3
        vinsertf32x4 $3, 64(%rcx,%rax), %zmm24, %zmm17          #47.3
        vinsertf32x4 $3, 80(%rcx,%rax), %zmm5, %zmm25           #47.3
        vshufps   $78, %zmm13, %zmm19, %zmm3                    #44.3
        vshufps   $228, %zmm13, %zmm20, %zmm4                   #44.3
        vpermi2ps %zmm17, %zmm10, %zmm21                        #47.3
        vpermt2ps %zmm25, %zmm11, %zmm10                        #47.3
        vpermt2ps %zmm25, %zmm28, %zmm17                        #47.3
        movslq    8(%rdi), %r9                                  #51.16
        lea       (%r9,%r9,2), %r8                              #51.8
        shlq      $5, %r8                                       #51.8
        prefetcht0 (%rcx,%r8)                                   #52.3
        movl      $23055, %r9d                                  #64.3
        movslq    8(%rsi), %r10                                 #53.16
        kmovw     %r9d, %k1                                     #64.3
        movl      $42480, %r9d                                  #64.3
        kmovw     %r9d, %k2                                     #64.3
        movl      $38595, %r9d                                  #83.3
        lea       (%r10,%r10,2), %rax                           #53.8
        shlq      $5, %rax                                      #53.8
        kmovw     %r9d, %k3                                     #83.3
        movl      $26940, %r9d                                  #83.3
        kmovw     %r9d, %k4                                     #83.3
        prefetcht0 (%rcx,%rax)                                  #54.3
        movslq    12(%rdi), %rdi                                #55.16
        lea       (%rdi,%rdi,2), %r9                            #55.9
        shlq      $5, %r9                                       #55.9
        prefetcht0 (%rcx,%r9)                                   #56.3
        movslq    12(%rsi), %rsi                                #57.16
        lea       (%rsi,%rsi,2), %rdi                           #57.9
        shlq      $5, %rdi                                      #57.9
        prefetcht0 (%rcx,%rdi)                                  #58.3
        vmovups   .L_2il0floatpacket.9(%rip), %zmm11            #64.3
        vmovups   (%rdx), %zmm31                                #68.3
        vmovups   144(%rdx), %zmm0                              #68.3
        vmovups   .L_2il0floatpacket.12(%rip), %zmm12           #68.3
        vmovups   .L_2il0floatpacket.11(%rip), %zmm7            #68.3
        vmovups   .L_2il0floatpacket.15(%rip), %zmm15           #68.3
        vmovups   .L_2il0floatpacket.14(%rip), %zmm14           #68.3
        vmovups   .L_2il0floatpacket.13(%rip), %zmm13           #68.3
        vmovups   64(%rdx), %zmm5                               #68.3
        vmovups   208(%rdx), %zmm6                              #68.3
        vpermps   %zmm10, %zmm11, %zmm16                        #65.3
        vpermps   %zmm21, %zmm11, %zmm22                        #64.3
        vpermps   %zmm17, %zmm11, %zmm8                         #66.3
        vaddps    %zmm16, %zmm3, %zmm3{%k1}                     #65.3
        vaddps    %zmm22, %zmm27, %zmm27{%k1}                   #64.3
        vaddps    %zmm8, %zmm4, %zmm4{%k1}                      #66.3
        vsubps    %zmm16, %zmm3, %zmm3{%k2}                     #65.3
        vsubps    %zmm22, %zmm27, %zmm27{%k2}                   #64.3
        vsubps    %zmm8, %zmm4, %zmm4{%k2}                      #66.3
        vmovups   .L_2il0floatpacket.10(%rip), %zmm10           #68.3
        vmovups   .L_2il0floatpacket.17(%rip), %zmm17           #68.3
        vmovups   .L_2il0floatpacket.16(%rip), %zmm16           #68.3
        vmovups   .L_2il0floatpacket.18(%rip), %zmm8            #68.3
        vmovaps   %zmm31, %zmm26                                #68.3
        vpermt2ps 72(%rdx), %zmm10, %zmm26                      #68.3
        vmovaps   %zmm0, %zmm24                                 #68.3
        vpermt2ps 216(%rdx), %zmm10, %zmm24                     #68.3
        vmovaps   %zmm26, %zmm18                                #68.3
        vpermt2ps %zmm24, %zmm12, %zmm18                        #68.3
        vmulps    %zmm18, %zmm3, %zmm28                         #68.3
        vmovups   .L_2il0floatpacket.19(%rip), %zmm18           #68.3
        vmovaps   %zmm26, %zmm9                                 #68.3
        vpermt2ps %zmm24, %zmm7, %zmm9                          #68.3
        vpermilps $177, %zmm3, %zmm23                           #68.3
        vmulps    %zmm23, %zmm15, %zmm1                         #68.3
        vmulps    %zmm9, %zmm27, %zmm25                         #68.3
        vmovups   .L_2il0floatpacket.20(%rip), %zmm9            #68.3
        vmovaps   %zmm26, %zmm2                                 #68.3
        vpermt2ps %zmm24, %zmm14, %zmm2                         #68.3
        vpermt2ps 72(%rdx), %zmm9, %zmm31                       #68.3
        vpermt2ps 216(%rdx), %zmm9, %zmm0                       #68.3
        vfmadd231ps %zmm27, %zmm2, %zmm28                       #68.3
        vpermilps $177, %zmm27, %zmm20                          #68.3
        vmulps    %zmm15, %zmm20, %zmm30                        #68.3
        vmovups   .L_2il0floatpacket.22(%rip), %zmm20           #68.3
        vmovaps   %zmm26, %zmm21                                #68.3
        vpermt2ps %zmm24, %zmm17, %zmm21                        #68.3
        vmovaps   %zmm26, %zmm19                                #68.3
        vpermt2ps %zmm24, %zmm13, %zmm19                        #68.3
        vfmadd231ps %zmm1, %zmm21, %zmm28                       #68.3
        vmovups   .L_2il0floatpacket.23(%rip), %zmm21           #68.3
        vfmadd231ps %zmm3, %zmm19, %zmm25                       #68.3
        vmovups   .L_2il0floatpacket.21(%rip), %zmm19           #68.3
        vmovaps   %zmm26, %zmm29                                #68.3
        vmovaps   %zmm26, %zmm22                                #68.3
        vpermt2ps %zmm24, %zmm18, %zmm26                        #68.3
        vpermt2ps %zmm24, %zmm16, %zmm29                        #68.3
        vpermt2ps %zmm24, %zmm8, %zmm22                         #68.3
        vfmadd231ps %zmm30, %zmm26, %zmm28                      #68.3
        vfmadd231ps %zmm30, %zmm29, %zmm25                      #68.3
        vmovaps   %zmm31, %zmm26                                #68.3
        vpermt2ps %zmm0, %zmm20, %zmm26                         #68.3
        vfmadd231ps %zmm1, %zmm22, %zmm25                       #68.3
        vmovups   .L_2il0floatpacket.24(%rip), %zmm22           #68.3
        vmulps    %zmm26, %zmm27, %zmm26                        #68.3
        vmovaps   %zmm31, %zmm27                                #68.3
        vmovaps   %zmm31, %zmm24                                #68.3
        vpermt2ps %zmm0, %zmm21, %zmm27                         #68.3
        vpermt2ps %zmm0, %zmm19, %zmm24                         #68.3
        vfmadd231ps %zmm4, %zmm27, %zmm28                       #68.3
        vfmadd231ps %zmm4, %zmm24, %zmm25                       #68.3
        vpermilps $177, %zmm4, %zmm27                           #68.3
        vmovaps   %zmm31, %zmm24                                #68.3
        vmulps    %zmm27, %zmm15, %zmm23                        #68.3
        vmovups   .L_2il0floatpacket.27(%rip), %zmm27           #68.3
        vpermt2ps %zmm0, %zmm22, %zmm24                         #68.3
        vfmadd213ps %zmm26, %zmm24, %zmm3                       #68.3
        vmovups   .L_2il0floatpacket.25(%rip), %zmm24           #68.3
        vmovups   .L_2il0floatpacket.26(%rip), %zmm26           #68.3
        vmovaps   %zmm31, %zmm29                                #68.3
        vmovaps   %zmm31, %zmm2                                 #68.3
        vpermt2ps %zmm0, %zmm24, %zmm29                         #68.3
        vpermt2ps %zmm0, %zmm26, %zmm2                          #68.3
        vfmadd231ps %zmm23, %zmm29, %zmm25                      #68.3
        vmovups   .L_2il0floatpacket.28(%rip), %zmm29           #68.3
        vfmadd213ps %zmm3, %zmm2, %zmm30                        #68.3
        vmovups   .L_2il0floatpacket.29(%rip), %zmm2            #68.3
        vmovaps   %zmm31, %zmm3                                 #68.3
        vpermt2ps %zmm0, %zmm27, %zmm3                          #68.3
        vpermt2ps %zmm0, %zmm29, %zmm31                         #68.3
        vpermt2ps 136(%rdx), %zmm2, %zmm5                       #68.3
        vpermt2ps 280(%rdx), %zmm2, %zmm6                       #68.3
        vfmadd231ps %zmm23, %zmm3, %zmm28                       #68.3
        vmovups   .L_2il0floatpacket.30(%rip), %zmm3            #68.3
        vfmadd213ps %zmm30, %zmm31, %zmm1                       #68.3
        vmovups   .L_2il0floatpacket.32(%rip), %ymm0            #70.3
        vmovups   .L_2il0floatpacket.33(%rip), %ymm31           #70.3
        vmovaps   %zmm5, %zmm30                                 #68.3
        vpermt2ps %zmm6, %zmm3, %zmm30                          #68.3
        vfmadd213ps %zmm1, %zmm30, %zmm4                        #68.3
        vmovups   .L_2il0floatpacket.31(%rip), %zmm1            #68.3
        vpermt2ps %zmm6, %zmm1, %zmm5                           #68.3
        vextractf64x4 $1, %zmm25, %ymm6                         #70.3
        vfmadd213ps %zmm4, %zmm5, %zmm23                        #68.3
        vmovaps   %zmm25, %zmm5                                 #70.3
        vpermps   %ymm5, %ymm0, %ymm4                           #70.3
        vpermps   %ymm6, %ymm0, %ymm25                          #70.3
        vfmadd213ps %ymm4, %ymm31, %ymm5                        #70.3
        vfmadd213ps %ymm6, %ymm31, %ymm25                       #70.3
        vmovups   .L_2il0floatpacket.34(%rip), %ymm6            #70.3
        vmulps    gamma_f(%rip){1to8}, %ymm5, %ymm30            #70.3
        vpermilps %ymm6, %ymm25, %ymm4                          #70.3
        vmovups   .L_2il0floatpacket.35(%rip), %ymm25           #70.3
        vfmadd213ps %ymm30, %ymm25, %ymm4                       #70.3
        vmovups   %ymm4, -112(%rbp)                             #70.3[spill]
        vmovaps   %zmm28, %zmm4                                 #71.3
        vpermps   %ymm4, %ymm0, %ymm30                          #71.3
        vfmadd213ps %ymm30, %ymm31, %ymm4                       #71.3
        vextractf64x4 $1, %zmm28, %ymm28                        #71.3
        vmulps    gamma_f(%rip){1to8}, %ymm4, %ymm5             #71.3
        vpermps   %ymm28, %ymm0, %ymm4                          #71.3
        vfmadd213ps %ymm28, %ymm31, %ymm4                       #71.3
        vpermilps %ymm6, %ymm4, %ymm30                          #71.3
        vfmadd213ps %ymm5, %ymm25, %ymm30                       #71.3
        vmovups   %ymm30, -80(%rbp)                             #71.3[spill]
        vmovups   32(%rcx,%r8), %xmm30                          #76.3
        vmovaps   %zmm23, %zmm4                                 #72.3
        vpermps   %ymm4, %ymm0, %ymm5                           #72.3
        vfmadd213ps %ymm5, %ymm31, %ymm4                        #72.3
        vextractf64x4 $1, %zmm23, %ymm23                        #72.3
        vmulps    gamma_f(%rip){1to8}, %ymm4, %ymm28            #72.3
        vpermps   %ymm23, %ymm0, %ymm4                          #72.3
        vfmadd213ps %ymm23, %ymm31, %ymm4                       #72.3
        vpermilps %ymm6, %ymm4, %ymm31                          #72.3
        vmovups   (%rcx,%r8), %xmm4                             #76.3
        vfmadd213ps %ymm28, %ymm25, %ymm31                      #72.3
        vmovups   16(%rcx,%r8), %xmm25                          #76.3
        vmovups   %ymm31, -48(%rbp)                             #72.3[spill]
        vinsertf32x4 $1, (%rcx,%rax), %zmm4, %zmm5              #76.3
        vinsertf32x4 $2, (%rcx,%r9), %zmm5, %zmm28              #76.3
        vinsertf32x4 $3, (%rcx,%rdi), %zmm28, %zmm4             #76.3
        vinsertf32x4 $1, 16(%rcx,%rax), %zmm25, %zmm23          #76.3
        vinsertf32x4 $1, 32(%rcx,%rax), %zmm30, %zmm5           #76.3
        vinsertf32x4 $2, 16(%rcx,%r9), %zmm23, %zmm6            #76.3
        vinsertf32x4 $2, 32(%rcx,%r9), %zmm5, %zmm28            #76.3
        vinsertf32x4 $3, 16(%rcx,%rdi), %zmm6, %zmm31           #76.3
        vinsertf32x4 $3, 32(%rcx,%rdi), %zmm28, %zmm25          #76.3
        vmovups   64(%rcx,%r8), %xmm6                           #79.3
        vshufps   $228, %zmm31, %zmm4, %zmm23                   #76.3
        vshufps   $228, %zmm25, %zmm31, %zmm5                   #76.3
        vmovups   48(%rcx,%r8), %xmm31                          #79.3
        vshufps   $78, %zmm25, %zmm4, %zmm4                     #76.3
        vinsertf32x4 $1, 48(%rcx,%rax), %zmm31, %zmm28          #79.3
        vmovups   80(%rcx,%r8), %xmm31                          #79.3
        vinsertf32x4 $2, 48(%rcx,%r9), %zmm28, %zmm25           #79.3
        vinsertf32x4 $3, 48(%rcx,%rdi), %zmm25, %zmm28          #79.3
        vinsertf32x4 $1, 64(%rcx,%rax), %zmm6, %zmm30           #79.3
        vinsertf32x4 $2, 64(%rcx,%r9), %zmm30, %zmm25           #79.3
        vinsertf32x4 $3, 64(%rcx,%rdi), %zmm25, %zmm30          #79.3
        vinsertf32x4 $1, 80(%rcx,%rax), %zmm31, %zmm25          #79.3
        vinsertf32x4 $2, 80(%rcx,%r9), %zmm25, %zmm6            #79.3
        vmovups   .L_2il0floatpacket.36(%rip), %zmm25           #79.3
        vinsertf32x4 $3, 80(%rcx,%rdi), %zmm6, %zmm6            #79.3
        vmovaps   %zmm28, %zmm31                                #79.3
        vpermt2ps %zmm30, %zmm25, %zmm31                        #79.3
        vpermt2ps %zmm6, %zmm25, %zmm30                         #79.3
        vpermps   %zmm31, %zmm11, %zmm25                        #83.3
        vpermps   %zmm30, %zmm11, %zmm30                        #85.3
        vaddps    %zmm25, %zmm23, %zmm23{%k3}                   #83.3
        vaddps    %zmm30, %zmm5, %zmm5{%k3}                     #85.3
        vsubps    %zmm25, %zmm23, %zmm23{%k4}                   #83.3
        vsubps    %zmm30, %zmm5, %zmm5{%k4}                     #85.3
        vmovups   .L_2il0floatpacket.37(%rip), %zmm25           #79.3
        vpermt2ps %zmm6, %zmm25, %zmm28                         #79.3
        vmovups   288(%rdx), %zmm25                             #91.3
        vmovups   496(%rdx), %zmm6                              #91.3
        vpermps   %zmm28, %zmm11, %zmm11                        #84.3
        vmovups   352(%rdx), %zmm28                             #91.3
        vaddps    %zmm11, %zmm4, %zmm4{%k3}                     #84.3
        vpermt2ps 568(%rdx), %zmm2, %zmm6                       #91.3
        vpermt2ps 424(%rdx), %zmm2, %zmm28                      #91.3
        vsubps    %zmm11, %zmm4, %zmm4{%k4}                     #84.3
        vmovups   432(%rdx), %zmm11                             #91.3
        vpermi2ps %zmm6, %zmm28, %zmm3                          #91.3
        vpermt2ps %zmm6, %zmm1, %zmm28                          #91.3
        vmovaps   %zmm25, %zmm2                                 #91.3
        vpermt2ps 360(%rdx), %zmm10, %zmm2                      #91.3
        vpermi2ps 504(%rdx), %zmm11, %zmm10                     #91.3
        vpermt2ps 504(%rdx), %zmm9, %zmm11                      #91.3
        vpermt2ps 360(%rdx), %zmm9, %zmm25                      #91.3
        vpermi2ps %zmm10, %zmm2, %zmm7                          #91.3
        vpermi2ps %zmm10, %zmm2, %zmm12                         #91.3
        vpermi2ps %zmm10, %zmm2, %zmm13                         #91.3
        vpermi2ps %zmm10, %zmm2, %zmm14                         #91.3
        vpermi2ps %zmm10, %zmm2, %zmm17                         #91.3
        vpermi2ps %zmm10, %zmm2, %zmm16                         #91.3
        vpermi2ps %zmm10, %zmm2, %zmm8                          #91.3
        vpermt2ps %zmm10, %zmm18, %zmm2                         #91.3
        vpermi2ps %zmm11, %zmm25, %zmm20                        #91.3
        vpermi2ps %zmm11, %zmm25, %zmm22                        #91.3
        vpermi2ps %zmm11, %zmm25, %zmm26                        #91.3
        vpermi2ps %zmm11, %zmm25, %zmm19                        #91.3
        vpermi2ps %zmm11, %zmm25, %zmm21                        #91.3
        vpermi2ps %zmm11, %zmm25, %zmm24                        #91.3
        vpermi2ps %zmm11, %zmm25, %zmm27                        #91.3
        vpermt2ps %zmm11, %zmm29, %zmm25                        #91.3
        vmulps    %zmm7, %zmm23, %zmm7                          #91.3
        vmulps    %zmm12, %zmm4, %zmm12                         #91.3
        vfmadd231ps %zmm4, %zmm13, %zmm7                        #91.3
        vfmadd231ps %zmm23, %zmm14, %zmm12                      #91.3
        vpermilps $177, %zmm4, %zmm13                           #91.3
        vmulps    %zmm13, %zmm15, %zmm6                         #91.3
        vpermilps $177, %zmm23, %zmm1                           #91.3
        vmulps    %zmm15, %zmm1, %zmm1                          #91.3
        vfmadd231ps %zmm6, %zmm17, %zmm12                       #91.3
        vfmadd231ps %zmm1, %zmm16, %zmm7                        #91.3
        vfmadd231ps %zmm1, %zmm2, %zmm12                        #91.3
        vmulps    %zmm20, %zmm23, %zmm2                         #91.3
        vfmadd231ps %zmm6, %zmm8, %zmm7                         #91.3
        vfmadd231ps %zmm5, %zmm21, %zmm12                       #91.3
        vmovups   .L_2il0floatpacket.33(%rip), %ymm21           #97.3
        vmovups   .L_2il0floatpacket.39(%rip), %ymm20           #97.3
        vfmadd213ps %zmm2, %zmm22, %zmm4                        #91.3
        vfmadd231ps %zmm5, %zmm19, %zmm7                        #91.3
        vmovups   .L_2il0floatpacket.38(%rip), %ymm19           #97.3
        vmovups   .L_2il0floatpacket.40(%rip), %ymm23           #97.3
        vfmadd213ps %zmm4, %zmm26, %zmm1                        #91.3
        vpermilps $177, %zmm5, %zmm8                            #91.3
        vmulps    %zmm8, %zmm15, %zmm15                         #91.3
        vfmadd213ps %zmm1, %zmm25, %zmm6                        #91.3
        vfmadd231ps %zmm15, %zmm24, %zmm7                       #91.3
        vfmadd213ps %zmm6, %zmm3, %zmm5                         #91.3
        vfmadd231ps %zmm15, %zmm27, %zmm12                      #91.3
        vmovups   .L_2il0floatpacket.41(%rip), %ymm24           #97.3
        vmovss    one_over_gammaf(%rip), %xmm3                  #94.8
        vmulss    coe(%rip), %xmm3, %xmm4                       #94.24
        vfmadd213ps %zmm5, %zmm28, %zmm15                       #91.3
        vmovss    %xmm4, -16(%rbp)                              #94.3
        vbroadcastss -16(%rbp), %ymm28                          #95.12
        vpermps   %ymm7, %ymm0, %ymm5                           #97.3
        vpermps   %ymm12, %ymm0, %ymm14                         #98.3
        vpermps   %ymm15, %ymm0, %ymm18                         #99.3
        vfmadd213ps %ymm7, %ymm21, %ymm5                        #97.3
        vfmadd213ps %ymm12, %ymm21, %ymm14                      #98.3
        vfmadd213ps %ymm15, %ymm21, %ymm18                      #99.3
        vextractf64x4 $1, %zmm7, %ymm9                          #97.3
        vextractf64x4 $1, %zmm12, %ymm13                        #98.3
        vextractf64x4 $1, %zmm15, %ymm22                        #99.3
        vpermps   %ymm9, %ymm0, %ymm10                          #97.3
        vpermps   %ymm13, %ymm0, %ymm16                         #98.3
        vpermps   %ymm22, %ymm0, %ymm0                          #99.3
        vpermilps %ymm19, %ymm5, %ymm11                         #97.3
        vfmadd213ps %ymm9, %ymm21, %ymm10                       #97.3
        vfmadd213ps %ymm13, %ymm21, %ymm16                      #98.3
        vfmadd213ps %ymm22, %ymm21, %ymm0                       #99.3
        vfmadd213ps -112(%rbp), %ymm20, %ymm11                  #97.3[spill]
        vpermilps %ymm19, %ymm14, %ymm17                        #98.3
        vpermilps %ymm19, %ymm18, %ymm25                        #99.3
        vfmadd213ps -80(%rbp), %ymm20, %ymm17                   #98.3[spill]
        vfmadd213ps -48(%rbp), %ymm20, %ymm25                   #99.3[spill]
        vpermilps %ymm23, %ymm10, %ymm26                        #97.3
        vpermilps %ymm23, %ymm16, %ymm27                        #98.3
        vpermilps %ymm23, %ymm0, %ymm29                         #99.3
        vfmadd213ps %ymm11, %ymm24, %ymm26                      #97.3
        vfmadd213ps %ymm17, %ymm24, %ymm27                      #98.3
        vfmadd213ps %ymm25, %ymm24, %ymm29                      #99.3
        vmulps    %ymm28, %ymm26, %ymm31                        #101.8
        vmulps    %ymm27, %ymm28, %ymm1                         #102.8
        vmulps    %ymm29, %ymm28, %ymm2                         #103.8
        vshufps   $68, %ymm1, %ymm31, %ymm30                    #105.3
        vshufps   $228, %ymm31, %ymm2, %ymm0                    #105.3
        vmovups   %xmm30, rs(%rip)                              #105.3
        vextractf32x4 $1, %ymm30, 48+rs(%rip)                   #105.3
        vmovups   %xmm0, 16+rs(%rip)                            #105.3
        vextractf128 $1, %ymm0, 64+rs(%rip)                     #105.3
                                # LOE r12 r13 r14 r15 ymm1 ymm2
..B1.4:                         # Preds ..B1.1
                                # Execution count [1.00e+00]
        vshufps   $238, %ymm2, %ymm1, %ymm0                     #105.3
        vmovups   %xmm0, 32+rs(%rip)                            #105.3
        vextractf128 $1, %ymm0, 80+rs(%rip)                     #105.3
        vzeroupper                                              #106.1
        movq      %rbp, %rsp                                    #106.1
        popq      %rbp                                          #106.1
	.cfi_restore 6
        movq      %rbx, %rsp                                    #106.1
        popq      %rbx                                          #106.1
	.cfi_def_cfa 7, 8
	.cfi_restore 3
        ret                                                     #106.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	doe_avx512,@function
	.size	doe_avx512,.-doe_avx512
	.data
# -- End  doe_avx512
	.text
# -- Begin  deo_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl deo_avx512
# --- deo_avx512(int *, int *, su3 *, spinor *)
deo_avx512:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_deo_avx512.11:
..L12:
                                                         #109.1
        movq      %rsi, %r9                                     #109.1
        movslq    (%rdi), %r10                                  #123.16
        lea       (%r10,%r10,2), %r8                            #123.8
        shlq      $5, %r8                                       #123.8
        prefetcht0 (%rcx,%r8)                                   #124.3
        movl      $42255, %esi                                  #171.3
        movslq    (%r9), %r11                                   #125.16
        kmovw     %esi, %k1                                     #171.3
        movl      $23280, %esi                                  #171.3
        kmovw     %esi, %k2                                     #171.3
        movl      $38595, %esi                                  #189.3
        lea       (%r11,%r11,2), %rax                           #125.8
        shlq      $5, %rax                                      #125.8
        kmovw     %esi, %k3                                     #189.3
        movl      $26940, %esi                                  #189.3
        kmovw     %esi, %k4                                     #189.3
        prefetcht0 (%rcx,%rax)                                  #126.3
        movslq    4(%rdi), %rsi                                 #127.17
        lea       (%rsi,%rsi,2), %r11                           #127.9
        shlq      $5, %r11                                      #127.9
        prefetcht0 (%rcx,%r11)                                  #128.3
        movslq    4(%r9), %r10                                  #129.17
        lea       (%r10,%r10,2), %rsi                           #129.9
        shlq      $5, %rsi                                      #129.9
        prefetcht0 (%rcx,%rsi)                                  #130.3
        vmovups   rs(%rip), %xmm0                               #132.3
        vmovups   16+rs(%rip), %xmm2                            #132.3
        vbroadcastss ceo(%rip), %ymm27                          #134.10
        vmovups   32+rs(%rip), %xmm6                            #132.3
        vbroadcastss one_over_gammaf(%rip), %ymm16              #143.10
        vmovups   .L_2il0floatpacket.32(%rip), %ymm23           #139.3
        vmovups   .L_2il0floatpacket.43(%rip), %ymm5            #148.3
        vmovups   .L_2il0floatpacket.33(%rip), %ymm28           #139.3
        vmovups   .L_2il0floatpacket.44(%rip), %ymm7            #148.3
        vmovups   216(%rdx), %zmm9                              #156.3
        vmovups   .L_2il0floatpacket.12(%rip), %zmm26           #156.3
        vinsertf128 $1, 48+rs(%rip), %ymm0, %ymm4               #132.3
        vinsertf128 $1, 64+rs(%rip), %ymm2, %ymm15              #132.3
        vshufps   $228, %ymm15, %ymm4, %ymm21                   #132.3
        vmulps    %ymm27, %ymm21, %ymm21                        #135.8
        vmulps    %ymm16, %ymm21, %ymm20                        #144.8
        vpermps   %ymm21, %ymm23, %ymm24                        #139.3
        vfmadd231ps %ymm21, %ymm28, %ymm24                      #139.3
        vmovups   .L_2il0floatpacket.11(%rip), %zmm21           #156.3
        vinsertf128 $1, 80+rs(%rip), %ymm6, %ymm13              #132.3
        vshufps   $78, %ymm13, %ymm4, %ymm30                    #132.3
        vshufps   $228, %ymm13, %ymm15, %ymm31                  #132.3
        vmulps    %ymm30, %ymm27, %ymm15                        #136.8
        vmovups   .L_2il0floatpacket.42(%rip), %ymm30           #148.3
        vmulps    %ymm31, %ymm27, %ymm17                        #137.8
        vmulps    %ymm15, %ymm16, %ymm25                        #145.8
        vmulps    %ymm17, %ymm16, %ymm14                        #146.8
        vpermps   %ymm20, %ymm30, %ymm1                         #148.3
        vpermps   %ymm20, %ymm5, %ymm13                         #148.3
        vfmadd213ps %ymm13, %ymm7, %ymm1                        #148.3
        vmovups   280(%rdx), %zmm13                             #156.3
        vpermps   %ymm15, %ymm23, %ymm22                        #140.3
        vfmadd231ps %ymm15, %ymm28, %ymm22                      #140.3
        vmovups   72(%rdx), %zmm15                              #156.3
        vpermps   %ymm25, %ymm30, %ymm12                        #149.3
        vpermps   %ymm25, %ymm5, %ymm18                         #149.3
        vpermps   %ymm14, %ymm30, %ymm11                        #150.3
        vpermps   %ymm14, %ymm5, %ymm0                          #150.3
        vfmadd213ps %ymm18, %ymm7, %ymm12                       #149.3
        vfmadd213ps %ymm0, %ymm7, %ymm11                        #150.3
        vmovups   .L_2il0floatpacket.15(%rip), %zmm30           #156.3
        vmovaps   %zmm15, %zmm7                                 #156.3
        vpermps   %ymm17, %ymm23, %ymm29                        #141.3
        vfmadd231ps %ymm17, %ymm28, %ymm29                      #141.3
        vmovups   .L_2il0floatpacket.22(%rip), %zmm28           #156.3
        vmovups   136(%rdx), %zmm17                             #156.3
        vinsertf64x4 $1, %ymm1, %zmm24, %zmm10                  #148.3
        vmovups   .L_2il0floatpacket.10(%rip), %zmm24           #156.3
        vpermt2ps (%rdx), %zmm24, %zmm7                         #156.3
        vmovaps   %zmm9, %zmm1                                  #156.3
        vpermt2ps 144(%rdx), %zmm24, %zmm1                      #156.3
        vmovaps   %zmm7, %zmm27                                 #156.3
        vpermt2ps %zmm1, %zmm26, %zmm27                         #156.3
        vmovaps   %zmm7, %zmm8                                  #156.3
        vpermt2ps %zmm1, %zmm21, %zmm8                          #156.3
        vmulps    %zmm8, %zmm10, %zmm18                         #156.3
        vmovups   .L_2il0floatpacket.13(%rip), %zmm8            #156.3
        vpermilps $177, %zmm10, %zmm6                           #156.3
        vmovaps   %zmm7, %zmm2                                  #156.3
        vmulps    %zmm30, %zmm6, %zmm5                          #156.3
        vmovups   .L_2il0floatpacket.17(%rip), %zmm6            #156.3
        vmovaps   %zmm7, %zmm31                                 #156.3
        vpermt2ps %zmm1, %zmm8, %zmm31                          #156.3
        vmovaps   %zmm7, %zmm23                                 #156.3
        vmovaps   %zmm7, %zmm0                                  #156.3
        vpermt2ps %zmm1, %zmm6, %zmm0                           #156.3
        vinsertf64x4 $1, %ymm12, %zmm22, %zmm3                  #149.3
        vmovups   .L_2il0floatpacket.18(%rip), %zmm22           #156.3
        vmulps    %zmm27, %zmm3, %zmm19                         #156.3
        vfmadd231ps %zmm3, %zmm31, %zmm18                       #156.3
        vmovups   .L_2il0floatpacket.23(%rip), %zmm27           #156.3
        vmovups   .L_2il0floatpacket.24(%rip), %zmm31           #156.3
        vpermilps $177, %zmm3, %zmm4                            #156.3
        vmulps    %zmm4, %zmm30, %zmm16                         #156.3
        vmovups   .L_2il0floatpacket.19(%rip), %zmm4            #156.3
        vinsertf64x4 $1, %ymm11, %zmm29, %zmm12                 #150.3
        vmovups   .L_2il0floatpacket.14(%rip), %zmm29           #156.3
        vpermt2ps %zmm1, %zmm29, %zmm2                          #156.3
        vfmadd231ps %zmm10, %zmm2, %zmm19                       #156.3
        vmovups   .L_2il0floatpacket.16(%rip), %zmm2            #156.3
        vfmadd231ps %zmm16, %zmm0, %zmm19                       #156.3
        vpermt2ps %zmm1, %zmm2, %zmm23                          #156.3
        vfmadd231ps %zmm5, %zmm23, %zmm18                       #156.3
        vmovups   .L_2il0floatpacket.20(%rip), %zmm23           #156.3
        vpermt2ps (%rdx), %zmm23, %zmm15                        #156.3
        vpermt2ps 144(%rdx), %zmm23, %zmm9                      #156.3
        vmovaps   %zmm15, %zmm0                                 #156.3
        vpermt2ps %zmm9, %zmm28, %zmm0                          #156.3
        vmovaps   %zmm7, %zmm11                                 #156.3
        vpermt2ps %zmm1, %zmm4, %zmm7                           #156.3
        vpermt2ps %zmm1, %zmm22, %zmm11                         #156.3
        vmovups   .L_2il0floatpacket.21(%rip), %zmm1            #156.3
        vfmadd231ps %zmm5, %zmm7, %zmm19                        #156.3
        vfmadd231ps %zmm16, %zmm11, %zmm18                      #156.3
        vmulps    %zmm0, %zmm10, %zmm0                          #156.3
        vmovaps   %zmm15, %zmm10                                #156.3
        vmovaps   %zmm15, %zmm7                                 #156.3
        vpermt2ps %zmm9, %zmm27, %zmm10                         #156.3
        vpermt2ps %zmm9, %zmm1, %zmm7                           #156.3
        vfmadd231ps %zmm12, %zmm10, %zmm19                      #156.3
        vfmadd231ps %zmm12, %zmm7, %zmm18                       #156.3
        vpermilps $177, %zmm12, %zmm10                          #156.3
        vmovaps   %zmm15, %zmm7                                 #156.3
        vmulps    %zmm10, %zmm30, %zmm11                        #156.3
        vmovups   .L_2il0floatpacket.26(%rip), %zmm10           #156.3
        vpermt2ps %zmm9, %zmm31, %zmm7                          #156.3
        vfmadd213ps %zmm0, %zmm7, %zmm3                         #156.3
        vmovups   .L_2il0floatpacket.25(%rip), %zmm7            #156.3
        vmovaps   %zmm15, %zmm0                                 #156.3
        vpermt2ps %zmm9, %zmm7, %zmm0                           #156.3
        vfmadd231ps %zmm11, %zmm0, %zmm18                       #156.3
        vmovaps   %zmm15, %zmm0                                 #156.3
        vpermt2ps %zmm9, %zmm10, %zmm0                          #156.3
        vfmadd213ps %zmm3, %zmm0, %zmm5                         #156.3
        vmovups   .L_2il0floatpacket.27(%rip), %zmm3            #156.3
        vmovaps   %zmm15, %zmm0                                 #156.3
        vpermt2ps %zmm9, %zmm3, %zmm0                           #156.3
        vfmadd231ps %zmm11, %zmm0, %zmm19                       #156.3
        vmovups   .L_2il0floatpacket.28(%rip), %zmm0            #156.3
        vpermt2ps %zmm9, %zmm0, %zmm15                          #156.3
        vmovups   .L_2il0floatpacket.29(%rip), %zmm9            #156.3
        vfmadd213ps %zmm5, %zmm15, %zmm16                       #156.3
        vmovups   .L_2il0floatpacket.30(%rip), %zmm5            #156.3
        vpermt2ps 64(%rdx), %zmm9, %zmm17                       #156.3
        vpermt2ps 208(%rdx), %zmm9, %zmm13                      #156.3
        vmovaps   %zmm17, %zmm15                                #156.3
        vpermt2ps %zmm13, %zmm5, %zmm15                         #156.3
        vfmadd213ps %zmm16, %zmm15, %zmm12                      #156.3
        vmovups   .L_2il0floatpacket.31(%rip), %zmm16           #156.3
        vmovups   16(%rcx,%rax), %xmm15                         #158.3
        vpermt2ps %zmm13, %zmm16, %zmm17                        #156.3
        vfmadd213ps %zmm12, %zmm17, %zmm11                      #156.3
        vmovups   (%rcx,%rax), %xmm17                           #158.3
        vinsertf32x4 $1, (%rcx,%r8), %zmm17, %zmm13             #158.3
        vinsertf32x4 $2, (%rcx,%rsi), %zmm13, %zmm12            #158.3
        vinsertf32x4 $3, (%rcx,%r11), %zmm12, %zmm13            #158.3
        vmovups   32(%rcx,%rax), %xmm12                         #158.3
        vinsertf32x4 $1, 16(%rcx,%r8), %zmm15, %zmm16           #158.3
        vinsertf32x4 $2, 16(%rcx,%rsi), %zmm16, %zmm17          #158.3
        vinsertf32x4 $3, 16(%rcx,%r11), %zmm17, %zmm17          #158.3
        vinsertf32x4 $1, 32(%rcx,%r8), %zmm12, %zmm15           #158.3
        vinsertf32x4 $2, 32(%rcx,%rsi), %zmm15, %zmm16          #158.3
        vshufps   $228, %zmm17, %zmm13, %zmm15                  #158.3
        vinsertf32x4 $3, 32(%rcx,%r11), %zmm16, %zmm12          #158.3
        vshufps   $78, %zmm12, %zmm13, %zmm16                   #158.3
        vshufps   $228, %zmm12, %zmm17, %zmm13                  #158.3
        vaddps    %zmm18, %zmm15, %zmm12                        #161.8
        vaddps    %zmm19, %zmm16, %zmm17                        #162.8
        vaddps    %zmm11, %zmm13, %zmm13                        #163.8
        vshufps   $68, %zmm17, %zmm12, %zmm15                   #164.3
        vshufps   $228, %zmm12, %zmm13, %zmm16                  #164.3
        vshufps   $238, %zmm13, %zmm17, %zmm17                  #164.3
        vmovups   %xmm15, (%rcx,%rax)                           #164.3
        vextractf32x4 $1, %zmm15, (%rcx,%r8)                    #164.3
        vextractf32x4 $2, %zmm15, (%rcx,%rsi)                   #164.3
        vextractf32x4 $3, %zmm15, (%rcx,%r11)                   #164.3
        vmovups   %xmm16, 16(%rcx,%rax)                         #164.3
        vextractf32x4 $1, %zmm16, 16(%rcx,%r8)                  #164.3
        vextractf32x4 $2, %zmm16, 16(%rcx,%rsi)                 #164.3
        vextractf32x4 $3, %zmm16, 16(%rcx,%r11)                 #164.3
        vmovups   %xmm17, 32(%rcx,%rax)                         #164.3
        vextractf32x4 $1, %zmm17, 32(%rcx,%r8)                  #164.3
        vextractf32x4 $2, %zmm17, 32(%rcx,%rsi)                 #164.3
        vextractf32x4 $3, %zmm17, 32(%rcx,%r11)                 #164.3
        vmovups   48(%rcx,%rax), %xmm12                         #168.3
        vmovups   64(%rcx,%rax), %xmm13                         #168.3
        vmovups   80(%rcx,%rax), %xmm17                         #168.3
        vinsertf32x4 $1, 48(%rcx,%r8), %zmm12, %zmm15           #168.3
        vinsertf32x4 $2, 48(%rcx,%rsi), %zmm15, %zmm16          #168.3
        vinsertf32x4 $3, 48(%rcx,%r11), %zmm16, %zmm12          #168.3
        vinsertf32x4 $1, 64(%rcx,%r8), %zmm13, %zmm15           #168.3
        vinsertf32x4 $2, 64(%rcx,%rsi), %zmm15, %zmm16          #168.3
        vinsertf32x4 $3, 64(%rcx,%r11), %zmm16, %zmm13          #168.3
        vinsertf32x4 $1, 80(%rcx,%r8), %zmm17, %zmm15           #168.3
        vinsertf32x4 $2, 80(%rcx,%rsi), %zmm15, %zmm16          #168.3
        vinsertf32x4 $3, 80(%rcx,%r11), %zmm16, %zmm17          #168.3
        vmovups   .L_2il0floatpacket.7(%rip), %zmm16            #168.3
        vmovaps   %zmm12, %zmm15                                #168.3
        vpermt2ps %zmm13, %zmm16, %zmm15                        #168.3
        vpermt2ps %zmm17, %zmm16, %zmm13                        #168.3
        vmovups   .L_2il0floatpacket.8(%rip), %zmm16            #168.3
        vpermt2ps %zmm17, %zmm16, %zmm12                        #168.3
        vmovups   .L_2il0floatpacket.9(%rip), %zmm16            #171.3
        vpermps   %zmm18, %zmm16, %zmm18                        #171.3
        vpermps   %zmm11, %zmm16, %zmm17                        #173.3
        vpermps   %zmm19, %zmm16, %zmm19                        #172.3
        vaddps    %zmm18, %zmm15, %zmm15{%k1}                   #171.3
        vaddps    %zmm17, %zmm13, %zmm13{%k1}                   #173.3
        vaddps    %zmm19, %zmm12, %zmm12{%k1}                   #172.3
        vsubps    %zmm18, %zmm15, %zmm15{%k2}                   #171.3
        vsubps    %zmm17, %zmm13, %zmm13{%k2}                   #173.3
        vsubps    %zmm19, %zmm12, %zmm12{%k2}                   #172.3
        vmovups   .L_2il0floatpacket.46(%rip), %zmm11           #174.3
        vmovups   .L_2il0floatpacket.45(%rip), %zmm17           #174.3
        vpermi2ps %zmm15, %zmm13, %zmm11                        #174.3
        vmovaps   %zmm15, %zmm18                                #174.3
        vmovups   .L_2il0floatpacket.47(%rip), %zmm15           #174.3
        vpermt2ps %zmm12, %zmm17, %zmm18                        #174.3
        vpermt2ps %zmm13, %zmm15, %zmm12                        #174.3
        vmovups   %xmm18, 48(%rcx,%rax)                         #174.3
        vextractf32x4 $1, %zmm18, 48(%rcx,%r8)                  #174.3
        vextractf32x4 $2, %zmm18, 48(%rcx,%rsi)                 #174.3
        vextractf32x4 $3, %zmm18, 48(%rcx,%r11)                 #174.3
        vmovups   %xmm11, 64(%rcx,%rax)                         #174.3
        vextractf32x4 $1, %zmm11, 64(%rcx,%r8)                  #174.3
        vextractf32x4 $2, %zmm11, 64(%rcx,%rsi)                 #174.3
        vextractf32x4 $3, %zmm11, 64(%rcx,%r11)                 #174.3
        vmovups   %xmm12, 80(%rcx,%rax)                         #174.3
        vextractf32x4 $1, %zmm12, 80(%rcx,%r8)                  #174.3
        vextractf32x4 $2, %zmm12, 80(%rcx,%rsi)                 #174.3
        vextractf32x4 $3, %zmm12, 80(%rcx,%r11)                 #174.3
        movslq    8(%rdi), %rax                                 #180.16
        lea       (%rax,%rax,2), %rsi                           #180.8
        shlq      $5, %rsi                                      #180.8
        prefetcht0 (%rcx,%rsi)                                  #181.3
        movslq    8(%r9), %r8                                   #182.16
        lea       (%r8,%r8,2), %rax                             #182.8
        shlq      $5, %rax                                      #182.8
        prefetcht0 (%rcx,%rax)                                  #183.3
        movslq    12(%rdi), %rdi                                #184.17
        lea       (%rdi,%rdi,2), %r8                            #184.9
        shlq      $5, %r8                                       #184.9
        prefetcht0 (%rcx,%r8)                                   #185.3
        movslq    12(%r9), %r9                                  #186.17
        lea       (%r9,%r9,2), %rdi                             #186.9
        shlq      $5, %rdi                                      #186.9
        prefetcht0 (%rcx,%rdi)                                  #187.3
        vmovups   .L_2il0floatpacket.48(%rip), %zmm13           #189.3
        vmovups   .L_2il0floatpacket.49(%rip), %zmm18           #189.3
        vmovups   424(%rdx), %zmm12                             #197.3
        vpermps   %zmm20, %zmm13, %zmm11                        #189.3
        vpermps   %zmm20, %zmm18, %zmm20                        #189.3
        vpermt2ps 352(%rdx), %zmm9, %zmm12                      #197.3
        vaddps    %zmm11, %zmm20, %zmm19{%k3}{z}                #189.3
        vsubps    %zmm11, %zmm20, %zmm19{%k4}                   #189.3
        vpermps   %zmm25, %zmm13, %zmm20                        #190.3
        vpermps   %zmm25, %zmm18, %zmm25                        #190.3
        vpermps   %zmm14, %zmm13, %zmm13                        #191.3
        vpermps   %zmm14, %zmm18, %zmm14                        #191.3
        vaddps    %zmm20, %zmm25, %zmm11{%k3}{z}                #190.3
        vmovups   568(%rdx), %zmm18                             #197.3
        vsubps    %zmm20, %zmm25, %zmm11{%k4}                   #190.3
        vaddps    %zmm13, %zmm14, %zmm20{%k3}{z}                #191.3
        vpermt2ps 496(%rdx), %zmm9, %zmm18                      #197.3
        vmovups   360(%rdx), %zmm25                             #197.3
        vsubps    %zmm13, %zmm14, %zmm20{%k4}                   #191.3
        vpermi2ps %zmm18, %zmm12, %zmm5                         #197.3
        vmovups   504(%rdx), %zmm14                             #197.3
        vmovaps   %zmm25, %zmm9                                 #197.3
        vpermt2ps 288(%rdx), %zmm24, %zmm9                      #197.3
        vpermi2ps 432(%rdx), %zmm14, %zmm24                     #197.3
        vpermt2ps 432(%rdx), %zmm23, %zmm14                     #197.3
        vpermt2ps 288(%rdx), %zmm23, %zmm25                     #197.3
        vpermi2ps %zmm24, %zmm9, %zmm21                         #197.3
        vpermi2ps %zmm24, %zmm9, %zmm26                         #197.3
        vpermi2ps %zmm24, %zmm9, %zmm8                          #197.3
        vpermi2ps %zmm24, %zmm9, %zmm29                         #197.3
        vpermi2ps %zmm24, %zmm9, %zmm6                          #197.3
        vpermi2ps %zmm24, %zmm9, %zmm2                          #197.3
        vpermi2ps %zmm24, %zmm9, %zmm22                         #197.3
        vpermt2ps %zmm24, %zmm4, %zmm9                          #197.3
        vpermi2ps %zmm14, %zmm25, %zmm28                        #197.3
        vpermi2ps %zmm14, %zmm25, %zmm31                        #197.3
        vpermi2ps %zmm14, %zmm25, %zmm1                         #197.3
        vpermi2ps %zmm14, %zmm25, %zmm10                        #197.3
        vpermi2ps %zmm14, %zmm25, %zmm27                        #197.3
        vpermi2ps %zmm14, %zmm25, %zmm7                         #197.3
        vpermi2ps %zmm14, %zmm25, %zmm3                         #197.3
        vpermt2ps %zmm14, %zmm0, %zmm25                         #197.3
        vmovups   (%rcx,%rax), %xmm0                            #199.3
        vmulps    %zmm21, %zmm19, %zmm21                        #197.3
        vmulps    %zmm26, %zmm11, %zmm13                        #197.3
        vmovups   16(%rcx,%rax), %xmm4                          #199.3
        vfmadd231ps %zmm11, %zmm8, %zmm21                       #197.3
        vfmadd231ps %zmm19, %zmm29, %zmm13                      #197.3
        vpermilps $177, %zmm19, %zmm26                          #197.3
        vmulps    %zmm30, %zmm26, %zmm8                         #197.3
        vpermilps $177, %zmm11, %zmm26                          #197.3
        vmulps    %zmm26, %zmm30, %zmm26                        #197.3
        vfmadd231ps %zmm8, %zmm2, %zmm21                        #197.3
        vfmadd231ps %zmm26, %zmm6, %zmm13                       #197.3
        vfmadd231ps %zmm26, %zmm22, %zmm21                      #197.3
        vfmadd231ps %zmm8, %zmm9, %zmm13                        #197.3
        vfmadd231ps %zmm20, %zmm1, %zmm21                       #197.3
        vmulps    %zmm28, %zmm19, %zmm9                         #197.3
        vfmadd231ps %zmm20, %zmm27, %zmm13                      #197.3
        vfmadd213ps %zmm9, %zmm31, %zmm11                       #197.3
        vpermilps $177, %zmm20, %zmm28                          #197.3
        vmulps    %zmm28, %zmm30, %zmm1                         #197.3
        vfmadd213ps %zmm11, %zmm10, %zmm8                       #197.3
        vfmadd231ps %zmm1, %zmm3, %zmm13                        #197.3
        vfmadd213ps %zmm8, %zmm25, %zmm26                       #197.3
        vmovups   .L_2il0floatpacket.31(%rip), %zmm3            #197.3
        vfmadd231ps %zmm1, %zmm7, %zmm21                        #197.3
        vmovups   32(%rcx,%rax), %xmm7                          #199.3
        vfmadd213ps %zmm26, %zmm5, %zmm20                       #197.3
        vpermt2ps %zmm18, %zmm3, %zmm12                         #197.3
        vfmadd213ps %zmm20, %zmm12, %zmm1                       #197.3
        vinsertf32x4 $1, (%rcx,%rsi), %zmm0, %zmm2              #199.3
        vinsertf32x4 $1, 16(%rcx,%rsi), %zmm4, %zmm5            #199.3
        vinsertf32x4 $2, (%rcx,%rdi), %zmm2, %zmm3              #199.3
        vinsertf32x4 $2, 16(%rcx,%rdi), %zmm5, %zmm6            #199.3
        vmovups   32(%rcx,%rdi), %xmm2                          #199.3
        vinsertf32x4 $3, (%rcx,%r8), %zmm3, %zmm3               #199.3
        vinsertf32x4 $3, 16(%rcx,%r8), %zmm6, %zmm4             #199.3
        vinsertf32x4 $1, 32(%rcx,%rsi), %zmm7, %zmm0            #199.3
                                # LOE rax rcx rbx rbp rsi rdi r8 r12 r13 r14 r15 xmm2 zmm0 zmm1 zmm3 zmm4 zmm13 zmm15 zmm16 zmm17 zmm21
..B2.4:                         # Preds ..B2.1
                                # Execution count [1.00e+00]
        vshufps   $228, %zmm4, %zmm3, %zmm5                     #199.3
        movl      $27075, %edx                                  #206.3
        vmovups   .L_2il0floatpacket.36(%rip), %zmm27           #205.3
        vmovups   .L_2il0floatpacket.37(%rip), %zmm26           #205.3
        vinsertf32x4 $2, %xmm2, %zmm0, %zmm0                    #199.3
        vaddps    %zmm21, %zmm5, %zmm6                          #200.8
        vpermps   %zmm21, %zmm16, %zmm21                        #206.3
        vinsertf32x4 $3, 32(%rcx,%r8), %zmm0, %zmm2             #199.3
        kmovw     %edx, %k1                                     #206.3
        vshufps   $78, %zmm2, %zmm3, %zmm3                      #199.3
        vshufps   $228, %zmm2, %zmm4, %zmm4                     #199.3
        vaddps    %zmm13, %zmm3, %zmm7                          #201.8
        vaddps    %zmm1, %zmm4, %zmm8                           #202.8
        vpermps   %zmm13, %zmm16, %zmm13                        #207.3
        vpermps   %zmm1, %zmm16, %zmm1                          #208.3
        vshufps   $68, %zmm7, %zmm6, %zmm9                      #203.3
        vshufps   $228, %zmm6, %zmm8, %zmm10                    #203.3
        vshufps   $238, %zmm8, %zmm7, %zmm11                    #203.3
        vmovups   .L_2il0floatpacket.50(%rip), %zmm16           #209.3
        vmovaps   %zmm27, %zmm29                                #205.3
        movl      $38460, %edx                                  #206.3
        kmovw     %edx, %k2                                     #206.3
        vmovups   %xmm9, (%rcx,%rax)                            #203.3
        vextractf32x4 $1, %zmm9, (%rcx,%rsi)                    #203.3
        vextractf32x4 $2, %zmm9, (%rcx,%rdi)                    #203.3
        vextractf32x4 $3, %zmm9, (%rcx,%r8)                     #203.3
        vmovups   %xmm10, 16(%rcx,%rax)                         #203.3
        vextractf32x4 $1, %zmm10, 16(%rcx,%rsi)                 #203.3
        vextractf32x4 $2, %zmm10, 16(%rcx,%rdi)                 #203.3
        vextractf32x4 $3, %zmm10, 16(%rcx,%r8)                  #203.3
        vmovups   %xmm11, 32(%rcx,%rax)                         #203.3
        vextractf32x4 $1, %zmm11, 32(%rcx,%rsi)                 #203.3
        vextractf32x4 $2, %zmm11, 32(%rcx,%rdi)                 #203.3
        vextractf32x4 $3, %zmm11, 32(%rcx,%r8)                  #203.3
        vmovups   48(%rcx,%rax), %xmm12                         #205.3
        vmovups   64(%rcx,%rax), %xmm19                         #205.3
        vmovups   80(%rcx,%rax), %xmm23                         #205.3
        vinsertf32x4 $1, 48(%rcx,%rsi), %zmm12, %zmm14          #205.3
        vinsertf32x4 $1, 64(%rcx,%rsi), %zmm19, %zmm20          #205.3
        vinsertf32x4 $1, 80(%rcx,%rsi), %zmm23, %zmm24          #205.3
        vinsertf32x4 $2, 48(%rcx,%rdi), %zmm14, %zmm18          #205.3
        vinsertf32x4 $2, 64(%rcx,%rdi), %zmm20, %zmm22          #205.3
        vinsertf32x4 $2, 80(%rcx,%rdi), %zmm24, %zmm25          #205.3
        vinsertf32x4 $3, 48(%rcx,%r8), %zmm18, %zmm31           #205.3
        vinsertf32x4 $3, 64(%rcx,%r8), %zmm22, %zmm30           #205.3
        vinsertf32x4 $3, 80(%rcx,%r8), %zmm25, %zmm28           #205.3
        vpermi2ps %zmm30, %zmm31, %zmm29                        #205.3
        vpermt2ps %zmm28, %zmm26, %zmm31                        #205.3
        vpermt2ps %zmm28, %zmm27, %zmm30                        #205.3
        vaddps    %zmm21, %zmm29, %zmm29{%k1}                   #206.3
        vaddps    %zmm13, %zmm31, %zmm31{%k1}                   #207.3
        vaddps    %zmm1, %zmm30, %zmm30{%k1}                    #208.3
        vsubps    %zmm21, %zmm29, %zmm29{%k2}                   #206.3
        vsubps    %zmm13, %zmm31, %zmm31{%k2}                   #207.3
        vsubps    %zmm1, %zmm30, %zmm30{%k2}                    #208.3
        vpermi2ps %zmm31, %zmm29, %zmm15                        #209.3
        vpermi2ps %zmm29, %zmm30, %zmm16                        #209.3
        vpermt2ps %zmm30, %zmm17, %zmm31                        #209.3
        vmovups   %xmm15, 48(%rcx,%rax)                         #209.3
        vextractf32x4 $1, %zmm15, 48(%rcx,%rsi)                 #209.3
        vextractf32x4 $2, %zmm15, 48(%rcx,%rdi)                 #209.3
        vextractf32x4 $3, %zmm15, 48(%rcx,%r8)                  #209.3
        vmovups   %xmm16, 64(%rcx,%rax)                         #209.3
        vextractf32x4 $1, %zmm16, 64(%rcx,%rsi)                 #209.3
        vextractf32x4 $2, %zmm16, 64(%rcx,%rdi)                 #209.3
        vextractf32x4 $3, %zmm16, 64(%rcx,%r8)                  #209.3
        vmovups   %xmm31, 80(%rcx,%rax)                         #209.3
        vextractf32x4 $1, %zmm31, 80(%rcx,%rsi)                 #209.3
        vextractf32x4 $2, %zmm31, 80(%rcx,%rdi)                 #209.3
        vextractf32x4 $3, %zmm31, 80(%rcx,%r8)                  #209.3
        vzeroupper                                              #210.1
        ret                                                     #210.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	deo_avx512,@function
	.size	deo_avx512,.-deo_avx512
	.data
# -- End  deo_avx512
	.section .rodata, "a"
	.align 64
	.align 64
.L_2il0floatpacket.7:
	.long	0x00000000,0x00000001,0x00000012,0x00000013,0x00000004,0x00000005,0x00000016,0x00000017,0x0000001a,0x0000001b,0x00000008,0x00000009,0x0000001e,0x0000001f,0x0000000c,0x0000000d
	.type	.L_2il0floatpacket.7,@object
	.size	.L_2il0floatpacket.7,64
	.align 64
.L_2il0floatpacket.8:
	.long	0x00000002,0x00000003,0x00000010,0x00000011,0x00000006,0x00000007,0x00000014,0x00000015,0x00000018,0x00000019,0x0000000a,0x0000000b,0x0000001c,0x0000001d,0x0000000e,0x0000000f
	.type	.L_2il0floatpacket.8,@object
	.size	.L_2il0floatpacket.8,64
	.align 64
.L_2il0floatpacket.9:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000004,0x00000005,0x00000006,0x00000007,0x00000009,0x00000008,0x0000000b,0x0000000a,0x0000000d,0x0000000c,0x0000000f,0x0000000e
	.type	.L_2il0floatpacket.9,@object
	.size	.L_2il0floatpacket.9,64
	.align 64
.L_2il0floatpacket.10:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000008,0x00000009,0x00000006,0x00000007,0x00000010,0x00000011,0x00000012,0x00000013,0x00000018,0x00000019,0x00000016,0x00000017
	.type	.L_2il0floatpacket.10,@object
	.size	.L_2il0floatpacket.10,64
	.align 64
.L_2il0floatpacket.11:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000008,0x00000008,0x00000008,0x00000008,0x00000010,0x00000010,0x00000010,0x00000010,0x00000018,0x00000018,0x00000018,0x00000018
	.type	.L_2il0floatpacket.11,@object
	.size	.L_2il0floatpacket.11,64
	.align 64
.L_2il0floatpacket.12:
	.long	0x00000004,0x00000004,0x00000004,0x00000004,0x0000000c,0x0000000c,0x0000000c,0x0000000c,0x00000014,0x00000014,0x00000014,0x00000014,0x0000001c,0x0000001c,0x0000001c,0x0000001c
	.type	.L_2il0floatpacket.12,@object
	.size	.L_2il0floatpacket.12,64
	.align 64
.L_2il0floatpacket.13:
	.long	0x00000002,0x00000002,0x00000002,0x00000002,0x0000000e,0x0000000e,0x0000000e,0x0000000e,0x00000012,0x00000012,0x00000012,0x00000012,0x0000001e,0x0000001e,0x0000001e,0x0000001e
	.type	.L_2il0floatpacket.13,@object
	.size	.L_2il0floatpacket.13,64
	.align 64
.L_2il0floatpacket.14:
	.long	0x00000006,0x00000006,0x00000006,0x00000006,0x0000000a,0x0000000a,0x0000000a,0x0000000a,0x00000016,0x00000016,0x00000016,0x00000016,0x0000001a,0x0000001a,0x0000001a,0x0000001a
	.type	.L_2il0floatpacket.14,@object
	.size	.L_2il0floatpacket.14,64
	.align 64
.L_2il0floatpacket.15:
	.long	0xbf800000,0x3f800000,0xbf800000,0x3f800000,0x3f800000,0xbf800000,0x3f800000,0xbf800000,0xbf800000,0x3f800000,0xbf800000,0x3f800000,0x3f800000,0xbf800000,0x3f800000,0xbf800000
	.type	.L_2il0floatpacket.15,@object
	.size	.L_2il0floatpacket.15,64
	.align 64
.L_2il0floatpacket.16:
	.long	0x00000001,0x00000001,0x00000001,0x00000001,0x00000009,0x00000009,0x00000009,0x00000009,0x00000011,0x00000011,0x00000011,0x00000011,0x00000019,0x00000019,0x00000019,0x00000019
	.type	.L_2il0floatpacket.16,@object
	.size	.L_2il0floatpacket.16,64
	.align 64
.L_2il0floatpacket.17:
	.long	0x00000005,0x00000005,0x00000005,0x00000005,0x0000000d,0x0000000d,0x0000000d,0x0000000d,0x00000015,0x00000015,0x00000015,0x00000015,0x0000001d,0x0000001d,0x0000001d,0x0000001d
	.type	.L_2il0floatpacket.17,@object
	.size	.L_2il0floatpacket.17,64
	.align 64
.L_2il0floatpacket.18:
	.long	0x00000003,0x00000003,0x00000003,0x00000003,0x0000000f,0x0000000f,0x0000000f,0x0000000f,0x00000013,0x00000013,0x00000013,0x00000013,0x0000001f,0x0000001f,0x0000001f,0x0000001f
	.type	.L_2il0floatpacket.18,@object
	.size	.L_2il0floatpacket.18,64
	.align 64
.L_2il0floatpacket.19:
	.long	0x00000007,0x00000007,0x00000007,0x00000007,0x0000000b,0x0000000b,0x0000000b,0x0000000b,0x00000017,0x00000017,0x00000017,0x00000017,0x0000001b,0x0000001b,0x0000001b,0x0000001b
	.type	.L_2il0floatpacket.19,@object
	.size	.L_2il0floatpacket.19,64
	.align 64
.L_2il0floatpacket.20:
	.long	0x00000004,0x00000005,0x0000000c,0x0000000d,0x0000000a,0x0000000b,0x0000000e,0x0000000f,0x00000014,0x00000015,0x0000001c,0x0000001d,0x0000001a,0x0000001b,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.20,@object
	.size	.L_2il0floatpacket.20,64
	.align 64
.L_2il0floatpacket.21:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x0000000a,0x0000000a,0x0000000a,0x0000000a,0x00000010,0x00000010,0x00000010,0x00000010,0x0000001a,0x0000001a,0x0000001a,0x0000001a
	.type	.L_2il0floatpacket.21,@object
	.size	.L_2il0floatpacket.21,64
	.align 64
.L_2il0floatpacket.22:
	.long	0x00000002,0x00000002,0x00000002,0x00000002,0x00000008,0x00000008,0x00000008,0x00000008,0x00000012,0x00000012,0x00000012,0x00000012,0x00000018,0x00000018,0x00000018,0x00000018
	.type	.L_2il0floatpacket.22,@object
	.size	.L_2il0floatpacket.22,64
	.align 64
.L_2il0floatpacket.23:
	.long	0x00000004,0x00000004,0x00000004,0x00000004,0x0000000e,0x0000000e,0x0000000e,0x0000000e,0x00000014,0x00000014,0x00000014,0x00000014,0x0000001e,0x0000001e,0x0000001e,0x0000001e
	.type	.L_2il0floatpacket.23,@object
	.size	.L_2il0floatpacket.23,64
	.align 64
.L_2il0floatpacket.24:
	.long	0x00000006,0x00000006,0x00000006,0x00000006,0x0000000c,0x0000000c,0x0000000c,0x0000000c,0x00000016,0x00000016,0x00000016,0x00000016,0x0000001c,0x0000001c,0x0000001c,0x0000001c
	.type	.L_2il0floatpacket.24,@object
	.size	.L_2il0floatpacket.24,64
	.align 64
.L_2il0floatpacket.25:
	.long	0x00000001,0x00000001,0x00000001,0x00000001,0x0000000b,0x0000000b,0x0000000b,0x0000000b,0x00000011,0x00000011,0x00000011,0x00000011,0x0000001b,0x0000001b,0x0000001b,0x0000001b
	.type	.L_2il0floatpacket.25,@object
	.size	.L_2il0floatpacket.25,64
	.align 64
.L_2il0floatpacket.26:
	.long	0x00000003,0x00000003,0x00000003,0x00000003,0x00000009,0x00000009,0x00000009,0x00000009,0x00000013,0x00000013,0x00000013,0x00000013,0x00000019,0x00000019,0x00000019,0x00000019
	.type	.L_2il0floatpacket.26,@object
	.size	.L_2il0floatpacket.26,64
	.align 64
.L_2il0floatpacket.27:
	.long	0x00000005,0x00000005,0x00000005,0x00000005,0x0000000f,0x0000000f,0x0000000f,0x0000000f,0x00000015,0x00000015,0x00000015,0x00000015,0x0000001f,0x0000001f,0x0000001f,0x0000001f
	.type	.L_2il0floatpacket.27,@object
	.size	.L_2il0floatpacket.27,64
	.align 64
.L_2il0floatpacket.28:
	.long	0x00000007,0x00000007,0x00000007,0x00000007,0x0000000d,0x0000000d,0x0000000d,0x0000000d,0x00000017,0x00000017,0x00000017,0x00000017,0x0000001d,0x0000001d,0x0000001d,0x0000001d
	.type	.L_2il0floatpacket.28,@object
	.size	.L_2il0floatpacket.28,64
	.align 64
.L_2il0floatpacket.29:
	.long	0x00000000,0x00000001,0x00000010,0x00000011,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.29,@object
	.size	.L_2il0floatpacket.29,64
	.align 64
.L_2il0floatpacket.30:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000002,0x00000002,0x00000002,0x00000002,0x00000010,0x00000010,0x00000010,0x00000010,0x00000012,0x00000012,0x00000012,0x00000012
	.type	.L_2il0floatpacket.30,@object
	.size	.L_2il0floatpacket.30,64
	.align 64
.L_2il0floatpacket.31:
	.long	0x00000001,0x00000001,0x00000001,0x00000001,0x00000003,0x00000003,0x00000003,0x00000003,0x00000011,0x00000011,0x00000011,0x00000011,0x00000013,0x00000013,0x00000013,0x00000013
	.type	.L_2il0floatpacket.31,@object
	.size	.L_2il0floatpacket.31,64
	.align 64
.L_2il0floatpacket.36:
	.long	0x00000012,0x00000013,0x00000000,0x00000001,0x00000016,0x00000017,0x00000004,0x00000005,0x00000008,0x00000009,0x0000001a,0x0000001b,0x0000000c,0x0000000d,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.36,@object
	.size	.L_2il0floatpacket.36,64
	.align 64
.L_2il0floatpacket.37:
	.long	0x00000010,0x00000011,0x00000002,0x00000003,0x00000014,0x00000015,0x00000006,0x00000007,0x0000000a,0x0000000b,0x00000018,0x00000019,0x0000000e,0x0000000f,0x0000001c,0x0000001d
	.type	.L_2il0floatpacket.37,@object
	.size	.L_2il0floatpacket.37,64
	.align 64
.L_2il0floatpacket.45:
	.long	0x00000000,0x00000001,0x00000010,0x00000011,0x00000004,0x00000005,0x00000014,0x00000015,0x0000000a,0x0000000b,0x0000001a,0x0000001b,0x0000000e,0x0000000f,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.45,@object
	.size	.L_2il0floatpacket.45,64
	.align 64
.L_2il0floatpacket.46:
	.long	0x00000000,0x00000001,0x00000012,0x00000013,0x00000004,0x00000005,0x00000016,0x00000017,0x0000000a,0x0000000b,0x00000018,0x00000019,0x0000000e,0x0000000f,0x0000001c,0x0000001d
	.type	.L_2il0floatpacket.46,@object
	.size	.L_2il0floatpacket.46,64
	.align 64
.L_2il0floatpacket.47:
	.long	0x00000002,0x00000003,0x00000012,0x00000013,0x00000006,0x00000007,0x00000016,0x00000017,0x00000008,0x00000009,0x00000018,0x00000019,0x0000000c,0x0000000d,0x0000001c,0x0000001d
	.type	.L_2il0floatpacket.47,@object
	.size	.L_2il0floatpacket.47,64
	.align 64
.L_2il0floatpacket.48:
	.long	0x00000006,0x00000007,0x00000004,0x00000005,0x00000006,0x00000007,0x00000004,0x00000005,0x00000005,0x00000004,0x00000007,0x00000006,0x00000005,0x00000004,0x00000007,0x00000006
	.type	.L_2il0floatpacket.48,@object
	.size	.L_2il0floatpacket.48,64
	.align 64
.L_2il0floatpacket.49:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000000,0x00000001,0x00000002,0x00000003,0x00000000,0x00000001,0x00000002,0x00000003,0x00000000,0x00000001,0x00000002,0x00000003
	.type	.L_2il0floatpacket.49,@object
	.size	.L_2il0floatpacket.49,64
	.align 64
.L_2il0floatpacket.50:
	.long	0x00000002,0x00000003,0x00000010,0x00000011,0x00000006,0x00000007,0x00000014,0x00000015,0x00000008,0x00000009,0x0000001a,0x0000001b,0x0000000c,0x0000000d,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.50,@object
	.size	.L_2il0floatpacket.50,64
	.align 32
.L_2il0floatpacket.32:
	.long	0x00000004,0x00000005,0x00000006,0x00000007,0x00000000,0x00000001,0x00000002,0x00000003
	.type	.L_2il0floatpacket.32,@object
	.size	.L_2il0floatpacket.32,32
	.align 32
.L_2il0floatpacket.33:
	.long	0x3f800000,0x3f800000,0x3f800000,0x3f800000,0xbf800000,0xbf800000,0xbf800000,0xbf800000
	.type	.L_2il0floatpacket.33,@object
	.size	.L_2il0floatpacket.33,32
	.align 32
.L_2il0floatpacket.34:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000003,0x00000002,0x00000001,0x00000000
	.type	.L_2il0floatpacket.34,@object
	.size	.L_2il0floatpacket.34,32
	.align 32
.L_2il0floatpacket.35:
	.long	0x3f800000,0x3f800000,0x3f800000,0x3f800000,0xbf800000,0x3f800000,0xbf800000,0x3f800000
	.type	.L_2il0floatpacket.35,@object
	.size	.L_2il0floatpacket.35,32
	.align 32
.L_2il0floatpacket.38:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000002,0x00000003,0x00000000,0x00000001
	.type	.L_2il0floatpacket.38,@object
	.size	.L_2il0floatpacket.38,32
	.align 32
.L_2il0floatpacket.39:
	.long	0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0xbf800000,0xbf800000
	.type	.L_2il0floatpacket.39,@object
	.size	.L_2il0floatpacket.39,32
	.align 32
.L_2il0floatpacket.40:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000001,0x00000000,0x00000003,0x00000002
	.type	.L_2il0floatpacket.40,@object
	.size	.L_2il0floatpacket.40,32
	.align 32
.L_2il0floatpacket.41:
	.long	0x3f800000,0x3f800000,0x3f800000,0x3f800000,0xbf800000,0x3f800000,0x3f800000,0xbf800000
	.type	.L_2il0floatpacket.41,@object
	.size	.L_2il0floatpacket.41,32
	.align 32
.L_2il0floatpacket.42:
	.long	0x00000007,0x00000006,0x00000005,0x00000004,0x00000007,0x00000006,0x00000005,0x00000004
	.type	.L_2il0floatpacket.42,@object
	.size	.L_2il0floatpacket.42,32
	.align 32
.L_2il0floatpacket.43:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000000,0x00000001,0x00000002,0x00000003
	.type	.L_2il0floatpacket.43,@object
	.size	.L_2il0floatpacket.43,32
	.align 32
.L_2il0floatpacket.44:
	.long	0xbf800000,0x3f800000,0xbf800000,0x3f800000,0x3f800000,0xbf800000,0x3f800000,0xbf800000
	.type	.L_2il0floatpacket.44,@object
	.size	.L_2il0floatpacket.44,32
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
