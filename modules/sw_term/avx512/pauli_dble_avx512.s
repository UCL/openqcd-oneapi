# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 17.0.4.196 Build 20170411";
# mark_description "-I../../..//include -I.. -I/cineca/prod/opt/compilers/intel/pe-xe-2017/binary/impi/2017.3.196/intel64/includ";
# mark_description "e -isystem /cineca/prod/opt/compilers/intel/pe-xe-2018/binary/impi/2018.1.163/include64/ -DNPROC0=1 -DNPROC1";
# mark_description "=1 -DNPROC2=1 -DNPROC3=1 -DL0=8 -DL1=8 -DL2=8 -DL3=8 -DNPROC0_BLK=1 -DNPROC1_BLK=1 -DNPROC2_BLK=1 -DNPROC3_B";
# mark_description "LK=1 -std=c89 -xCORE-AVX512 -mtune=skylake -DAVX512 -O3 -Ddirac_counters -pedantic -fstrict-aliasing -Wno-lo";
# mark_description "ng-long -Wstrict-prototypes -S";
	.file "pauli_dble_avx512.c"
	.text
..TXTST0:
# -- Begin  mul_pauli2_dble_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl mul_pauli2_dble_avx512
# --- mul_pauli2_dble_avx512(double, const pauli_dble *, const weyl_dble *, weyl_dble *)
mul_pauli2_dble_avx512:
# parameter 1: %xmm0
# parameter 2: %rdi
# parameter 3: %rsi
# parameter 4: %rdx
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_mul_pauli2_dble_avx512.1:
..L2:
                                                          #15.1
        pushq     %rbp                                          #15.1
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp                                    #15.1
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        movl      $86, %eax                                     #76.8
        vmovups   .L_2il0floatpacket.9(%rip), %zmm17            #44.9
        vmovups   .L_2il0floatpacket.10(%rip), %zmm9            #46.9
        vmovups   (%rsi), %zmm22                                #26.28
        vmovups   64(%rsi), %zmm16                              #27.28
        vmovups   128(%rsi), %zmm14                             #28.28
        vmovups   .L_2il0floatpacket.13(%rip), %zmm5            #65.9
        vmovups   (%rdi), %zmm19                                #30.27
        vmovups   64(%rdi), %zmm29                              #31.27
        vmovups   288(%rdi), %zmm21                             #35.27
        vmovups   352(%rdi), %zmm6                              #36.27
        vmovups   .L_2il0floatpacket.14(%rip), %zmm23           #68.9
        vmovsd    %xmm0, -16(%rbp)                              #15.1
        vmovups   .L_2il0floatpacket.16(%rip), %zmm8            #73.9
        vmovups   -16(%rbp), %zmm13                             #41.27
        vmovups   .L_2il0floatpacket.11(%rip), %zmm11           #49.8
        vmovups   .L_2il0floatpacket.12(%rip), %zmm20           #52.9
        vmovups   128(%rdi), %zmm30                             #32.27
        vmovups   416(%rdi), %zmm10                             #37.27
        vmovups   480(%rdi), %zmm7                              #38.27
        vmovups   192(%rdi), %zmm27                             #33.27
        vmovups   256(%rdi), %zmm28                             #34.27
        vpermi2pd %zmm6, %zmm29, %zmm23                         #68.9
        vpermi2pd %zmm14, %zmm22, %zmm11                        #49.8
        kmovw     %eax, %k1                                     #76.8
        vmovaps   %zmm17, %zmm2                                 #45.8
        movl      $169, %eax                                    #77.8
        vmovaps   %zmm9, %zmm1                                  #47.8
        vpermi2pd %zmm16, %zmm22, %zmm2                         #45.8
        vpermi2pd %zmm16, %zmm22, %zmm1                         #47.8
        vpermi2pd %zmm14, %zmm16, %zmm17                        #51.8
        vpermt2pd %zmm14, %zmm9, %zmm16                         #55.8
        vpermt2pd %zmm14, %zmm20, %zmm22                        #53.8
        vmovups   .L_2il0floatpacket.15(%rip), %zmm9            #70.9
        kmovw     %eax, %k2                                     #77.8
        vmovaps   %zmm5, %zmm25                                 #66.9
        movl      $106, %eax                                    #88.8
        vpermi2pd %zmm21, %zmm19, %zmm25                        #66.9
        kmovw     %eax, %k3                                     #88.8
        vmovaps   %zmm9, %zmm31                                 #71.10
        movl      $149, %eax                                    #89.8
        vmovaps   %zmm8, %zmm26                                 #74.10
        vpermi2pd %zmm23, %zmm25, %zmm31                        #71.10
        vpermi2pd %zmm23, %zmm13, %zmm26                        #74.10
        kmovw     %eax, %k4                                     #89.8
        vmulpd    %zmm2, %zmm31, %zmm24                         #72.8
        vmovups   .L_2il0floatpacket.18(%rip), %zmm31           #83.10
        vpermilpd $85, %zmm2, %zmm4                             #57.9
        movl      $85, %eax                                     #142.8
        vmulpd    %zmm4, %zmm26, %zmm12                         #75.10
        vmovups   .L_2il0floatpacket.17(%rip), %zmm26           #79.9
        vaddpd    %zmm12, %zmm24, %zmm24{%k1}                   #76.8
        vpermt2pd %zmm21, %zmm26, %zmm19                        #80.9
        vpermi2pd %zmm10, %zmm30, %zmm26                        #93.9
        vsubpd    %zmm12, %zmm24, %zmm24{%k2}                   #77.8
        vpermi2pd %zmm19, %zmm23, %zmm31                        #83.10
        vpermi2pd %zmm26, %zmm13, %zmm8                         #99.10
        vmovups   .L_2il0floatpacket.19(%rip), %zmm12           #85.9
        vfmadd213pd %zmm24, %zmm22, %zmm31                      #84.8
        vmovaps   %zmm12, %zmm21                                #86.10
        vpermi2pd %zmm13, %zmm23, %zmm21                        #86.10
        vpermi2pd %zmm13, %zmm26, %zmm12                        #108.10
        vpermilpd $85, %zmm22, %zmm15                           #61.9
        vmulpd    %zmm15, %zmm21, %zmm0                         #87.10
        vmovups   .L_2il0floatpacket.20(%rip), %zmm21           #96.10
        vaddpd    %zmm0, %zmm31, %zmm31{%k3}                    #88.8
        vpermi2pd %zmm26, %zmm25, %zmm21                        #96.10
        vsubpd    %zmm0, %zmm31, %zmm31{%k4}                    #89.8
        vmulpd    %zmm1, %zmm21, %zmm21                         #97.8
        vpermilpd $85, %zmm1, %zmm3                             #58.9
        vmulpd    %zmm3, %zmm8, %zmm24                          #100.10
        vmovups   .L_2il0floatpacket.21(%rip), %zmm8            #105.10
        vaddpd    %zmm24, %zmm21, %zmm21{%k1}                   #101.8
        vpermi2pd %zmm19, %zmm26, %zmm8                         #105.10
        vsubpd    %zmm24, %zmm21, %zmm21{%k2}                   #102.8
        vmovups   .L_2il0floatpacket.23(%rip), %zmm24           #118.10
        vfmadd213pd %zmm21, %zmm17, %zmm8                       #106.8
        vpermilpd $85, %zmm17, %zmm18                           #60.9
        vmulpd    %zmm18, %zmm12, %zmm21                        #109.10
        vmovups   .L_2il0floatpacket.24(%rip), %zmm12           #121.10
        vaddpd    %zmm21, %zmm8, %zmm8{%k3}                     #110.8
        vsubpd    %zmm21, %zmm8, %zmm8{%k4}                     #111.8
        vmovups   .L_2il0floatpacket.22(%rip), %zmm21           #114.9
        vmovaps   %zmm21, %zmm0                                 #115.9
        vpermi2pd %zmm7, %zmm27, %zmm0                          #115.9
        vpermi2pd %zmm0, %zmm19, %zmm24                         #118.10
        vpermi2pd %zmm0, %zmm13, %zmm12                         #121.10
        vmulpd    %zmm11, %zmm24, %zmm24                        #119.8
        vpermilpd $85, %zmm11, %zmm14                           #59.9
        vmulpd    %zmm14, %zmm12, %zmm12                        #122.10
        vaddpd    %zmm12, %zmm24, %zmm24{%k1}                   #123.8
        kmovw     %eax, %k1                                     #142.8
        vsubpd    %zmm12, %zmm24, %zmm24{%k2}                   #124.8
        vmovups   .L_2il0floatpacket.25(%rip), %zmm12           #127.10
        vpermi2pd %zmm19, %zmm0, %zmm12                         #127.10
        movl      $170, %eax                                    #143.8
        vmovups   .L_2il0floatpacket.26(%rip), %zmm19           #130.10
        vfmadd213pd %zmm24, %zmm16, %zmm12                      #128.8
        vpermi2pd %zmm13, %zmm0, %zmm19                         #130.10
        kmovw     %eax, %k7                                     #143.8
        vpermilpd $85, %zmm16, %zmm20                           #62.9
        movl      $90, %eax                                     #161.8
        vmulpd    %zmm20, %zmm19, %zmm13                        #131.10
        vmovups   .L_2il0floatpacket.27(%rip), %zmm19           #136.9
        kmovw     %eax, %k5                                     #161.8
        vaddpd    %zmm13, %zmm12, %zmm12{%k3}                   #132.8
        vsubpd    %zmm13, %zmm12, %zmm12{%k4}                   #133.8
        movl      $165, %eax                                    #162.8
        kmovw     %eax, %k6                                     #162.8
        vmovaps   %zmm19, %zmm13                                #137.10
        movl      $240, %eax                                    #267.8
        vpermi2pd %zmm23, %zmm25, %zmm13                        #137.10
        kmovw     %eax, %k2                                     #267.8
        vfmadd213pd %zmm8, %zmm2, %zmm13                        #138.8
        vmovups   .L_2il0floatpacket.28(%rip), %zmm8            #139.9
        vmovaps   %zmm8, %zmm24                                 #140.10
        vpermi2pd %zmm23, %zmm25, %zmm24                        #140.10
        vmulpd    %zmm24, %zmm4, %zmm24                         #141.10
        vaddpd    %zmm24, %zmm13, %zmm13{%k1}                   #142.8
        vsubpd    %zmm24, %zmm13, %zmm13{%k7}                   #143.8
        vmovaps   %zmm9, %zmm24                                 #146.10
        vpermi2pd %zmm0, %zmm23, %zmm24                         #146.10
        vfmadd213pd %zmm31, %zmm17, %zmm24                      #147.8
        vmovups   .L_2il0floatpacket.29(%rip), %zmm31           #148.9
        vpermt2pd %zmm0, %zmm31, %zmm23                         #149.10
        vmulpd    %zmm23, %zmm18, %zmm23                        #150.10
        vaddpd    %zmm23, %zmm24, %zmm24{%k7}                   #151.8
        vsubpd    %zmm23, %zmm24, %zmm24{%k1}                   #152.8
        vmovaps   %zmm19, %zmm23                                #156.10
        vpermi2pd %zmm26, %zmm25, %zmm23                        #156.10
        vpermt2pd %zmm26, %zmm8, %zmm25                         #159.10
        vfmadd213pd %zmm24, %zmm1, %zmm23                       #157.8
        vmulpd    %zmm25, %zmm3, %zmm25                         #160.10
        vaddpd    %zmm25, %zmm23, %zmm23{%k5}                   #161.8
        vsubpd    %zmm25, %zmm23, %zmm23{%k6}                   #162.8
        vmovaps   %zmm9, %zmm25                                 #165.10
        vpermi2pd %zmm0, %zmm26, %zmm25                         #165.10
        vpermt2pd %zmm0, %zmm31, %zmm26                         #168.10
        vfmadd213pd %zmm13, %zmm22, %zmm25                      #166.8
        vmulpd    %zmm26, %zmm15, %zmm26                        #169.10
        vaddpd    %zmm26, %zmm25, %zmm25{%k5}                   #170.8
        vsubpd    %zmm26, %zmm25, %zmm25{%k6}                   #171.8
        vmovups   .L_2il0floatpacket.30(%rip), %zmm26           #173.9
        vmovaps   %zmm26, %zmm0                                 #174.10
        vpermi2pd %zmm6, %zmm29, %zmm0                          #174.10
        vpermi2pd %zmm10, %zmm30, %zmm26                        #184.10
        vfmadd213pd %zmm12, %zmm2, %zmm0                        #175.8
        vmovups   .L_2il0floatpacket.31(%rip), %zmm12           #176.9
        vmovaps   %zmm12, %zmm2                                 #177.10
        vpermi2pd %zmm6, %zmm29, %zmm2                          #177.10
        vpermi2pd %zmm10, %zmm30, %zmm12                        #187.10
        vpermt2pd %zmm6, %zmm5, %zmm29                          #195.9
        vmulpd    %zmm2, %zmm4, %zmm4                           #178.10
        vaddpd    %zmm4, %zmm0, %zmm0{%k1}                      #179.8
        vsubpd    %zmm4, %zmm0, %zmm0{%k7}                      #180.8
        vfmadd213pd %zmm0, %zmm1, %zmm26                        #185.8
        vmulpd    %zmm12, %zmm3, %zmm1                          #188.10
        vmovups   .L_2il0floatpacket.32(%rip), %zmm3            #196.9
        vaddpd    %zmm1, %zmm26, %zmm26{%k1}                    #189.8
        vpermt2pd %zmm7, %zmm3, %zmm27                          #197.9
        vmovups   .L_2il0floatpacket.33(%rip), %zmm7            #209.9
        vsubpd    %zmm1, %zmm26, %zmm26{%k7}                    #190.8
        vpermt2pd 544(%rdi), %zmm7, %zmm28                      #210.9
        vmovaps   %zmm31, %zmm5                                 #203.10
        vpermi2pd %zmm27, %zmm29, %zmm5                         #203.10
        vmulpd    %zmm5, %zmm14, %zmm6                          #204.10
        vmovaps   %zmm9, %zmm0                                  #200.10
        vpermi2pd %zmm27, %zmm29, %zmm0                         #200.10
        vfmadd213pd %zmm23, %zmm11, %zmm0                       #201.8
        vaddpd    %zmm6, %zmm0, %zmm0{%k5}                      #205.8
        vmovaps   %zmm19, %zmm1                                 #231.10
        vsubpd    %zmm6, %zmm0, %zmm0{%k6}                      #206.8
        vpermi2pd %zmm28, %zmm29, %zmm1                         #231.10
        vpermt2pd %zmm28, %zmm8, %zmm29                         #234.10
        vfmadd213pd %zmm0, %zmm16, %zmm1                        #232.8
        vmovups   .L_2il0floatpacket.34(%rip), %zmm0            #240.9
        vmulpd    %zmm29, %zmm20, %zmm29                        #235.10
        vpermt2pd %zmm10, %zmm0, %zmm30                         #241.9
        vaddpd    %zmm29, %zmm1, %zmm1{%k7}                     #236.8
        vmovaps   %zmm19, %zmm5                                 #213.10
        vpermi2pd %zmm28, %zmm27, %zmm5                         #213.10
        vpermi2pd %zmm27, %zmm30, %zmm19                        #244.10
        vsubpd    %zmm29, %zmm1, %zmm1{%k1}                     #237.8
        vfmadd213pd %zmm26, %zmm22, %zmm5                       #214.8
        vfmadd213pd %zmm25, %zmm11, %zmm19                      #245.8
        vmovaps   %zmm8, %zmm22                                 #216.10
        vpermi2pd %zmm28, %zmm27, %zmm22                        #216.10
        vpermi2pd %zmm27, %zmm30, %zmm8                         #247.10
        vmulpd    %zmm22, %zmm15, %zmm15                        #217.10
        vmulpd    %zmm8, %zmm14, %zmm11                         #248.10
        vaddpd    %zmm15, %zmm5, %zmm5{%k5}                     #218.8
        vaddpd    %zmm11, %zmm19, %zmm19{%k5}                   #249.8
        vsubpd    %zmm15, %zmm5, %zmm5{%k6}                     #219.8
        vsubpd    %zmm11, %zmm19, %zmm19{%k6}                   #250.8
        vmovaps   %zmm31, %zmm6                                 #225.10
        vmovaps   %zmm9, %zmm15                                 #222.10
        vpermi2pd %zmm28, %zmm27, %zmm6                         #225.10
        vpermi2pd %zmm28, %zmm30, %zmm9                         #253.10
        vpermt2pd %zmm28, %zmm31, %zmm30                        #256.10
        vpermi2pd %zmm28, %zmm27, %zmm15                        #222.10
        vmulpd    %zmm6, %zmm18, %zmm18                         #226.10
        vfmadd213pd %zmm19, %zmm16, %zmm9                       #254.8
        vfmadd213pd %zmm5, %zmm17, %zmm15                       #223.8
        vmovups   .L_2il0floatpacket.35(%rip), %zmm28           #263.9
        vmulpd    %zmm30, %zmm20, %zmm16                        #257.10
        vaddpd    %zmm18, %zmm15, %zmm15{%k5}                   #227.8
        vaddpd    %zmm16, %zmm9, %zmm9{%k7}                     #258.8
        vsubpd    %zmm18, %zmm15, %zmm15{%k6}                   #228.8
        vsubpd    %zmm16, %zmm9, %zmm9{%k1}                     #259.8
        vpermi2pd %zmm1, %zmm15, %zmm28                         #264.8
        vpermi2pd %zmm9, %zmm1, %zmm7                           #262.8
        vpermt2pd %zmm15, %zmm21, %zmm9                         #266.8
        vblendmpd %zmm28, %zmm7, %zmm10{%k2}                    #267.8
        vblendmpd %zmm7, %zmm9, %zmm27{%k2}                     #268.8
        vblendmpd %zmm9, %zmm28, %zmm30{%k2}                    #269.8
        vmovups   %zmm10, (%rdx)                                #271.21
        vmovups   %zmm27, 64(%rdx)                              #272.21
        vmovups   %zmm30, 128(%rdx)                             #273.21
        vzeroupper                                              #274.1
        movq      %rbp, %rsp                                    #274.1
        popq      %rbp                                          #274.1
	.cfi_restore 6
        ret                                                     #274.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	mul_pauli2_dble_avx512,@function
	.size	mul_pauli2_dble_avx512,.-mul_pauli2_dble_avx512
	.data
# -- End  mul_pauli2_dble_avx512
	.text
# -- Begin  fwd_house_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl fwd_house_avx512
# --- fwd_house_avx512(double, complex_dble *, complex_dble *, double *)
fwd_house_avx512:
# parameter 1: %xmm0
# parameter 2: %rdi
# parameter 3: %rsi
# parameter 4: %rdx
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_fwd_house_avx512.8:
..L9:
                                                          #278.1
        pushq     %r12                                          #278.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #278.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #278.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #278.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rbx                                          #278.1
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
        pushq     %rbp                                          #278.1
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
        xorl      %eax, %eax                                    #283.3
        xorl      %r8d, %r8d                                    #285.3
        movq      %rdi, %r9                                     #278.1
        xorl      %r11d, %r11d                                  #285.3
        vmovapd   %xmm0, %xmm14                                 #278.1
        xorl      %r10d, %r10d                                  #285.3
        vxorpd    %xmm1, %xmm1, %xmm1                           #321.12
        vmovsd    .L_2il0floatpacket.38(%rip), %xmm11           #302.12
        xorl      %edi, %edi                                    #285.3
        vmovsd    .L_2il0floatpacket.36(%rip), %xmm0            #301.16
                                # LOE rdx rsi r8 r9 eax edi r10d r11d xmm0 xmm1 xmm11 xmm14
..B2.2:                         # Preds ..B2.35 ..B2.1
                                # Execution count [5.00e+00]
        movslq    %r10d, %r12                                   #287.29
        lea       1(%r8), %ecx                                  #290.10
        shlq      $4, %r12                                      #286.10
        vmovsd    8(%r9,%r12), %xmm3                            #287.29
        vmulsd    %xmm3, %xmm3, %xmm12                          #287.29
        vmovsd    (%r9,%r12), %xmm2                             #286.29
        vfmadd231sd %xmm2, %xmm2, %xmm12                        #286.5
        vsqrtsd   %xmm12, %xmm12, %xmm13                        #288.10
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.3:                         # Preds ..B2.2
                                # Execution count [5.00e+00]
        xorl      %r13d, %r13d                                  #290.5
        lea       5(%r11), %r14d                                #290.5
        movl      %r14d, %ebp                                   #290.5
        movl      $1, %ebx                                      #290.5
        sarl      $2, %ebp                                      #290.5
        shrl      $29, %ebp                                     #290.5
        lea       5(%rbp,%r11), %r15d                           #290.5
        xorl      %ebp, %ebp                                    #291.7
        sarl      $3, %r15d                                     #290.5
        testl     %r15d, %r15d                                  #290.5
        jbe       ..B2.7        # Prob 10%                      #290.5
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx ebp edi r10d r11d r13d r14d r15d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.4:                         # Preds ..B2.3
                                # Execution count [1.56e-02]
        vxorpd    %xmm10, %xmm10, %xmm10                        #290.5
        vxorpd    %xmm9, %xmm9, %xmm9                           #290.5
        vxorpd    %xmm8, %xmm8, %xmm8                           #290.5
        vxorpd    %xmm4, %xmm4, %xmm4                           #290.5
        vxorpd    %xmm7, %xmm7, %xmm7                           #290.5
        vxorpd    %xmm6, %xmm6, %xmm6                           #290.5
        vxorpd    %xmm5, %xmm5, %xmm5                           #290.5
                                # LOE rdx rsi r8 r9 r12 eax ecx ebp edi r10d r11d r13d r14d r15d xmm0 xmm1 xmm2 xmm3 xmm4 xmm5 xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14
..B2.5:                         # Preds ..B2.5 ..B2.4
                                # Execution count [3.12e+00]
        incl      %r13d                                         #290.5
        lea       (%r10,%rbp), %ebx                             #292.33
        movslq    %ebx, %rbx                                    #291.14
        addl      $48, %ebp                                     #290.5
        shlq      $4, %rbx                                      #292.33
        vmovsd    104(%r9,%rbx), %xmm15                         #292.14
        vmovsd    200(%r9,%rbx), %xmm18                         #292.14
        vmulsd    %xmm15, %xmm15, %xmm17                        #291.7
        vmulsd    %xmm18, %xmm18, %xmm20                        #291.7
        vmovsd    192(%r9,%rbx), %xmm19                         #291.14
        vmovsd    96(%r9,%rbx), %xmm16                          #291.14
        vfmadd231sd %xmm16, %xmm16, %xmm17                      #291.7
        vmovsd    296(%r9,%rbx), %xmm21                         #292.14
        vmovsd    392(%r9,%rbx), %xmm24                         #292.14
        vmovsd    488(%r9,%rbx), %xmm27                         #292.14
        vmovsd    584(%r9,%rbx), %xmm30                         #292.14
        vfmadd231sd %xmm19, %xmm19, %xmm20                      #291.7
        vaddsd    %xmm12, %xmm17, %xmm12                        #291.7
        vmulsd    %xmm21, %xmm21, %xmm23                        #291.7
        vmulsd    %xmm24, %xmm24, %xmm26                        #291.7
        vmulsd    %xmm27, %xmm27, %xmm29                        #291.7
        vaddsd    %xmm10, %xmm20, %xmm10                        #291.7
        vmulsd    %xmm30, %xmm30, %xmm15                        #291.7
        vmovsd    680(%r9,%rbx), %xmm16                         #292.14
        vmovsd    776(%r9,%rbx), %xmm19                         #292.14
        vmulsd    %xmm16, %xmm16, %xmm18                        #291.7
        vmulsd    %xmm19, %xmm19, %xmm21                        #291.7
        vmovsd    768(%r9,%rbx), %xmm20                         #291.14
        vmovsd    288(%r9,%rbx), %xmm22                         #291.14
        vmovsd    384(%r9,%rbx), %xmm25                         #291.14
        vmovsd    480(%r9,%rbx), %xmm28                         #291.14
        vmovsd    576(%r9,%rbx), %xmm31                         #291.14
        vmovsd    672(%r9,%rbx), %xmm17                         #291.14
        vfmadd231sd %xmm22, %xmm22, %xmm23                      #291.7
        vfmadd231sd %xmm25, %xmm25, %xmm26                      #291.7
        vfmadd231sd %xmm28, %xmm28, %xmm29                      #291.7
        vfmadd231sd %xmm31, %xmm31, %xmm15                      #291.7
        vfmadd231sd %xmm17, %xmm17, %xmm18                      #291.7
        vfmadd231sd %xmm20, %xmm20, %xmm21                      #291.7
        vaddsd    %xmm9, %xmm23, %xmm9                          #291.7
        vaddsd    %xmm8, %xmm26, %xmm8                          #291.7
        vaddsd    %xmm4, %xmm29, %xmm4                          #291.7
        vaddsd    %xmm7, %xmm15, %xmm7                          #291.7
        vaddsd    %xmm6, %xmm18, %xmm6                          #291.7
        vaddsd    %xmm5, %xmm21, %xmm5                          #291.7
        cmpl      %r15d, %r13d                                  #290.5
        jb        ..B2.5        # Prob 99%                      #290.5
                                # LOE rdx rsi r8 r9 r12 eax ecx ebp edi r10d r11d r13d r14d r15d xmm0 xmm1 xmm2 xmm3 xmm4 xmm5 xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14
..B2.6:                         # Preds ..B2.5
                                # Execution count [4.50e+00]
        vaddsd    %xmm10, %xmm12, %xmm10                        #290.5
        vaddsd    %xmm8, %xmm9, %xmm8                           #290.5
        vaddsd    %xmm7, %xmm4, %xmm4                           #290.5
        vaddsd    %xmm5, %xmm6, %xmm5                           #290.5
        vaddsd    %xmm8, %xmm10, %xmm9                          #290.5
        vaddsd    %xmm5, %xmm4, %xmm6                           #290.5
        vaddsd    %xmm6, %xmm9, %xmm12                          #290.5
        lea       1(,%r13,8), %ebx                              #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d r14d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.7:                         # Preds ..B2.6 ..B2.3
                                # Execution count [5.00e+00]
        cmpl      %r14d, %ebx                                   #290.5
        ja        ..B2.23       # Prob 50%                      #290.5
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.8:                         # Preds ..B2.7
                                # Execution count [0.00e+00]
        lea       (%r8,%rbx), %ebp                              #290.5
        negl      %ebp                                          #290.5
        addl      $5, %ebp                                      #290.5
        jmp       *.2.10_2.switchtab.4(,%rbp,8)                 #290.5
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.6:
..B2.10:                        # Preds ..B2.8
                                # Execution count [0.00e+00]
        lea       (%rbx,%rbx,2), %ebp                           #291.14
        lea       (%r10,%rbp,2), %r13d                          #292.33
        movslq    %r13d, %r13                                   #291.14
        shlq      $4, %r13                                      #292.33
        lea       584(%r9,%r13), %r14                           #292.14
        vmovsd    (%r14), %xmm4                                 #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    -8(%r14), %xmm5                               #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.5:
..B2.12:                        # Preds ..B2.8 ..B2.10
                                # Execution count [0.00e+00]
        lea       (%rbx,%rbx,2), %ebp                           #291.14
        lea       (%r10,%rbp,2), %r13d                          #292.33
        movslq    %r13d, %r13                                   #291.14
        shlq      $4, %r13                                      #292.33
        lea       488(%r9,%r13), %r14                           #292.14
        vmovsd    (%r14), %xmm4                                 #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    -8(%r14), %xmm5                               #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.4:
..B2.14:                        # Preds ..B2.8 ..B2.12
                                # Execution count [0.00e+00]
        lea       (%rbx,%rbx,2), %ebp                           #291.14
        lea       (%r10,%rbp,2), %r13d                          #292.33
        movslq    %r13d, %r13                                   #291.14
        shlq      $4, %r13                                      #292.33
        lea       392(%r9,%r13), %r14                           #292.14
        vmovsd    (%r14), %xmm4                                 #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    -8(%r14), %xmm5                               #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.3:
..B2.16:                        # Preds ..B2.8 ..B2.14
                                # Execution count [0.00e+00]
        lea       (%rbx,%rbx,2), %ebp                           #291.14
        lea       (%r10,%rbp,2), %r13d                          #292.33
        movslq    %r13d, %r13                                   #291.14
        shlq      $4, %r13                                      #292.33
        lea       296(%r9,%r13), %r14                           #292.14
        vmovsd    (%r14), %xmm4                                 #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    -8(%r14), %xmm5                               #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.2:
..B2.18:                        # Preds ..B2.8 ..B2.16
                                # Execution count [0.00e+00]
        lea       (%rbx,%rbx,2), %ebp                           #291.14
        lea       (%r10,%rbp,2), %r13d                          #292.33
        movslq    %r13d, %r13                                   #291.14
        shlq      $4, %r13                                      #292.33
        lea       200(%r9,%r13), %r14                           #292.14
        vmovsd    (%r14), %xmm4                                 #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    -8(%r14), %xmm5                               #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.1:
..B2.20:                        # Preds ..B2.8 ..B2.18
                                # Execution count [0.00e+00]
        lea       (%rbx,%rbx,2), %ebp                           #291.14
        lea       (%r10,%rbp,2), %r13d                          #292.33
        movslq    %r13d, %r13                                   #291.14
        shlq      $4, %r13                                      #292.33
        vmovsd    104(%r9,%r13), %xmm4                          #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    96(%r9,%r13), %xmm5                           #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx ebx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..1.10_0.TAG.0:
..B2.22:                        # Preds ..B2.8 ..B2.20
                                # Execution count [4.50e+00]
        lea       (%rbx,%rbx,2), %ebx                           #291.14
        lea       (%r10,%rbx,2), %ebp                           #292.33
        movslq    %ebp, %rbp                                    #291.14
        shlq      $4, %rbp                                      #292.33
        vmovsd    8(%r9,%rbp), %xmm4                            #292.14
        vmulsd    %xmm4, %xmm4, %xmm6                           #292.33
        vmovsd    (%r9,%rbp), %xmm5                             #291.14
        vfmadd231sd %xmm5, %xmm5, %xmm6                         #291.7
        vaddsd    %xmm6, %xmm12, %xmm12                         #291.7
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.23:                        # Preds ..B2.22 ..B2.7
                                # Execution count [5.00e+00]
        vcomisd   %xmm14, %xmm12                                #294.15
        jb        ..B2.25       # Prob 50%                      #294.15
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.24:                        # Preds ..B2.23
                                # Execution count [2.50e+00]
        vsqrtsd   %xmm12, %xmm12, %xmm12                        #295.12
        jmp       ..B2.26       # Prob 100%                     #295.12
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.25:                        # Preds ..B2.23
                                # Execution count [2.50e+00]
        vmovapd   %xmm11, %xmm12                                #298.7
        movl      $1, %eax                                      #297.7
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.26:                        # Preds ..B2.24 ..B2.25
                                # Execution count [5.00e+00]
        vmulsd    %xmm0, %xmm12, %xmm4                          #301.30
        vcomisd   %xmm4, %xmm13                                 #301.30
        jb        ..B2.28       # Prob 50%                      #301.30
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm11 xmm12 xmm13 xmm14
..B2.27:                        # Preds ..B2.26
                                # Execution count [2.50e+00]
        vdivsd    %xmm13, %xmm11, %xmm4                         #302.18
        vmulsd    %xmm4, %xmm2, %xmm5                           #303.19
        vmulsd    %xmm3, %xmm4, %xmm4                           #304.19
        jmp       ..B2.29       # Prob 100%                     #304.19
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm4 xmm5 xmm11 xmm12 xmm13 xmm14
..B2.28:                        # Preds ..B2.26
                                # Execution count [2.50e+00]
        vmovapd   %xmm11, %xmm5                                 #306.7
        vxorpd    %xmm4, %xmm4, %xmm4                           #307.7
                                # LOE rdx rsi r8 r9 r12 eax ecx edi r10d r11d xmm0 xmm1 xmm2 xmm3 xmm4 xmm5 xmm11 xmm12 xmm13 xmm14
..B2.29:                        # Preds ..B2.27 ..B2.28
                                # Execution count [6.63e-01]
        vfmadd231sd %xmm12, %xmm4, %xmm3                        #311.5
        xorl      %ebp, %ebp                                    #318.5
        vfmadd231sd %xmm12, %xmm5, %xmm2                        #310.5
        vmovsd    %xmm3, 8(%r9,%r12)                            #311.5
        vaddsd    %xmm13, %xmm12, %xmm3                         #313.28
        vmulsd    %xmm3, %xmm12, %xmm12                         #313.28
        lea       6(%r11), %r13d                                #326.23
        vmulsd    %xmm3, %xmm5, %xmm5                           #315.5
        vmulsd    %xmm3, %xmm4, %xmm4                           #316.28
        vdivsd    %xmm12, %xmm11, %xmm6                         #313.28
        vmovsd    %xmm2, (%r9,%r12)                             #310.5
        movq      %r8, %rbx                                     #315.5
        vmulsd    %xmm6, %xmm5, %xmm2                           #315.5
        vmulsd    %xmm6, %xmm4, %xmm8                           #316.33
        movslq    %edi, %r12                                    #323.12
        shlq      $4, %rbx                                      #315.5
        addq      %r8, %r12                                     #323.12
        shlq      $4, %r12                                      #323.12
        vxorpd    .L_2il0floatpacket.37(%rip), %xmm2, %xmm7     #315.5
        vmovsd    %xmm6, (%rdx,%r8,8)                           #314.5
        negq      %r8                                           #318.27
        vmovsd    %xmm6, -24(%rsp)                              #313.5
        addq      $5, %r8                                       #318.27
        vmovsd    %xmm7, (%rsi,%rbx)                            #315.5
        vmovsd    %xmm8, 8(%rsi,%rbx)                           #316.5
        lea       (%r9,%r12), %rbx                              #323.12
        vmovddup  -24(%rsp), %xmm2                              #338.28
        lea       16(%r12,%r9), %r12                            #324.12
        movq      %rdx, -16(%rsp)                               #326.23[spill]
                                # LOE rbx rbp rsi r8 r9 r12 eax ecx edi r10d r11d r13d xmm0 xmm1 xmm2 xmm11 xmm14
..B2.30:                        # Preds ..B2.34 ..B2.29
                                # Execution count [2.12e+01]
        vmovapd   %xmm1, %xmm3                                  #321.12
        movq      %rbx, %r15                                    #323.7
        movq      %r12, %r14                                    #324.7
        xorl      %edx, %edx                                    #326.7
                                # LOE rbx rbp rsi r8 r9 r12 r14 r15 eax edx ecx edi r10d r11d r13d xmm0 xmm1 xmm2 xmm3 xmm11 xmm14
..B2.31:                        # Preds ..B2.31 ..B2.30
                                # Execution count [1.18e+02]
        vmovupd   (%r14), %xmm5                                 #329.27
        incl      %edx                                          #326.7
        vmulpd    8(%r15){1to2}, %xmm5, %xmm4                   #330.14
        vpermilpd $1, %xmm4, %xmm6                              #331.14
        addq      $96, %r14                                     #335.9
        vfmsubadd231pd (%r15){1to2}, %xmm5, %xmm6               #332.14
        addq      $96, %r15                                     #334.9
        vaddpd    %xmm3, %xmm6, %xmm3                           #333.14
        cmpl      %r13d, %edx                                   #326.7
        jb        ..B2.31       # Prob 82%                      #326.7
                                # LOE rbx rbp rsi r8 r9 r12 r14 r15 eax edx ecx edi r10d r11d r13d xmm0 xmm1 xmm2 xmm3 xmm11 xmm14
..B2.32:                        # Preds ..B2.31
                                # Execution count [2.25e+01]
        vmulpd    %xmm2, %xmm3, %xmm3                           #339.12
        movq      %rbx, %r15                                    #342.7
        movq      %r12, %r14                                    #343.7
        xorl      %edx, %edx                                    #344.7
                                # LOE rbx rbp rsi r8 r9 r12 r14 r15 eax edx ecx edi r10d r11d r13d xmm0 xmm1 xmm2 xmm3 xmm11 xmm14
..B2.33:                        # Preds ..B2.33 ..B2.32
                                # Execution count [1.25e+02]
        vmulpd    8(%r15){1to2}, %xmm3, %xmm4                   #348.14
        vpermilpd $1, %xmm4, %xmm6                              #349.14
        incl      %edx                                          #344.7
        vfmaddsub231pd (%r15){1to2}, %xmm3, %xmm6               #350.14
        addq      $96, %r15                                     #353.9
        vmovupd   (%r14), %xmm5                                 #347.27
        vsubpd    %xmm6, %xmm5, %xmm7                           #351.14
        vmovupd   %xmm7, (%r14)                                 #352.24
        addq      $96, %r14                                     #354.9
        cmpl      %r13d, %edx                                   #344.7
        jb        ..B2.33       # Prob 82%                      #344.7
                                # LOE rbx rbp rsi r8 r9 r12 r14 r15 eax edx ecx edi r10d r11d r13d xmm0 xmm1 xmm2 xmm3 xmm11 xmm14
..B2.34:                        # Preds ..B2.33
                                # Execution count [2.50e+01]
        incq      %rbp                                          #318.5
        addq      $16, %r12                                     #318.5
        cmpq      %r8, %rbp                                     #318.5
        jb        ..B2.30       # Prob 81%                      #318.5
                                # LOE rbx rbp rsi r8 r9 r12 eax ecx edi r10d r11d r13d xmm0 xmm1 xmm2 xmm11 xmm14
..B2.35:                        # Preds ..B2.34
                                # Execution count [5.00e+00]
        decl      %r11d                                         #290.10
        addl      $7, %r10d                                     #290.10
        addl      $6, %edi                                      #290.10
        movl      %ecx, %r8d                                    #285.3
        movq      -16(%rsp), %rdx                               #[spill]
        cmpl      $5, %ecx                                      #285.3
        jb        ..B2.2        # Prob 79%                      #285.3
                                # LOE rdx rsi r8 r9 eax edi r10d r11d xmm0 xmm1 xmm11 xmm14
..B2.36:                        # Preds ..B2.35
                                # Execution count [1.00e+00]
        vmovsd    568(%r9), %xmm2                               #359.44
        vmulsd    %xmm2, %xmm2, %xmm0                           #359.44
        vmovsd    560(%r9), %xmm1                               #359.8
        vfmadd231sd %xmm1, %xmm1, %xmm0                         #359.3
        vcomisd   %xmm14, %xmm0                                 #361.13
        jb        ..B2.38       # Prob 50%                      #361.13
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1 xmm2 xmm11
..B2.37:                        # Preds ..B2.36
                                # Execution count [5.00e-01]
        vdivsd    %xmm0, %xmm11, %xmm11                         #362.16
        jmp       ..B2.39       # Prob 100%                     #362.16
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm1 xmm2 xmm11
..B2.38:                        # Preds ..B2.36
                                # Execution count [5.00e-01]
        movl      $1, %eax                                      #364.5
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm1 xmm2 xmm11
..B2.39:                        # Preds ..B2.37 ..B2.38
                                # Execution count [1.00e+00]
        vmulsd    %xmm11, %xmm1, %xmm0                          #368.19
        vmulsd    %xmm2, %xmm11, %xmm1                          #369.3
        vxorpd    .L_2il0floatpacket.37(%rip), %xmm1, %xmm2     #369.3
        vmovsd    %xmm0, 80(%rsi)                               #368.3
        vmovsd    %xmm2, 88(%rsi)                               #369.3
	.cfi_restore 6
        popq      %rbp                                          #371.10
	.cfi_def_cfa_offset 48
	.cfi_restore 3
        popq      %rbx                                          #371.10
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #371.10
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #371.10
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #371.10
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #371.10
	.cfi_def_cfa_offset 8
        ret                                                     #371.10
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	fwd_house_avx512,@function
	.size	fwd_house_avx512,.-fwd_house_avx512
	.section .rodata, "a"
	.align 64
	.align 8
.2.10_2.switchtab.4:
	.quad	..1.10_0.TAG.0
	.quad	..1.10_0.TAG.1
	.quad	..1.10_0.TAG.2
	.quad	..1.10_0.TAG.3
	.quad	..1.10_0.TAG.4
	.quad	..1.10_0.TAG.5
	.quad	..1.10_0.TAG.6
	.data
# -- End  fwd_house_avx512
	.text
# -- Begin  solv_sys_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl solv_sys_avx512
# --- solv_sys_avx512(complex_dble *, complex_dble *)
solv_sys_avx512:
# parameter 1: %rdi
# parameter 2: %rsi
..B3.1:                         # Preds ..B3.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_solv_sys_avx512.35:
..L36:
                                                         #376.1
        pushq     %r12                                          #376.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #376.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #376.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #376.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rbx                                          #376.1
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
        pushq     %rbp                                          #376.1
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
        movl      $5, %edx                                      #381.8
        vxorpd    %xmm0, %xmm0, %xmm0                           #405.24
        movl      $80, %eax                                     #381.8
                                # LOE rax rdx rsi rdi xmm0
..B3.2:                         # Preds ..B3.10 ..B3.1
                                # Execution count [5.00e+00]
        lea       -1(%rdx), %r13d                               #382.19
        movslq    %r13d, %r14                                   #382.10
        lea       -3(%rdx,%rdx,2), %ebp                         #382.10
        movq      %r14, %r12                                    #400.28
        addl      %ebp, %ebp                                    #382.10
        shlq      $4, %r12                                      #400.28
        movslq    %ebp, %rbp                                    #382.10
        addq      %rsi, %r12                                    #376.1
        shlq      $4, %rbp                                      #383.28
        testl     %r13d, %r13d                                  #382.28
        js        ..B3.10       # Prob 2%                       #382.28
                                # LOE rax rdx rbp rsi rdi r12 r14 r13d xmm0
..B3.3:                         # Preds ..B3.2
                                # Execution count [4.90e+00]
        lea       -1(%rdx), %r11                                #390.21
        movq      %r11, %rbx                                    #390.12
        lea       (%rdi,%rax), %r8                              #383.28
        shlq      $4, %rbx                                      #390.12
        lea       (%rbp,%r8), %r9                               #383.28
                                # LOE rax rdx rbx rbp rsi rdi r8 r9 r11 r12 r14 r13d xmm0
..B3.4:                         # Preds ..B3.8 ..B3.3
                                # Execution count [2.72e+01]
        vmovupd   (%rax,%rsi), %xmm2                            #385.25
        movq      %r11, %rcx                                    #390.12
        vmulpd    8(%r9){1to2}, %xmm2, %xmm1                    #386.12
        vpermilpd $1, %xmm1, %xmm1                              #387.12
        vfmaddsub231pd (%r9){1to2}, %xmm2, %xmm1                #388.12
        cmpq      %r14, %r11                                    #390.29
        jle       ..B3.8        # Prob 10%                      #390.29
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r11 r12 r14 r13d xmm0 xmm1
..B3.5:                         # Preds ..B3.4
                                # Execution count [2.45e+01]
        lea       (%rdi,%rbp), %r10                             #391.30
        addq      %rbx, %r10                                    #391.30
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r13d xmm0 xmm1
..B3.6:                         # Preds ..B3.6 ..B3.5
                                # Execution count [1.36e+02]
        lea       (%rcx,%rcx,2), %r15d                          #393.34
        addl      %r15d, %r15d                                  #393.34
        decq      %rcx                                          #390.32
        movslq    %r15d, %r15                                   #393.27
        shlq      $4, %r15                                      #393.27
        vmovupd   (%r8,%r15), %xmm3                             #393.27
        vmulpd    8(%r10){1to2}, %xmm3, %xmm2                   #394.14
        vpermilpd $1, %xmm2, %xmm4                              #395.14
        vfmaddsub231pd (%r10){1to2}, %xmm3, %xmm4               #396.14
        addq      $-16, %r10                                    #390.32
        vaddpd    %xmm4, %xmm1, %xmm1                           #397.14
        cmpq      %r14, %rcx                                    #390.29
        jg        ..B3.6        # Prob 82%                      #390.29
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r10 r11 r12 r14 r13d xmm0 xmm1
..B3.8:                         # Preds ..B3.6 ..B3.4
                                # Execution count [2.72e+01]
        vmulpd    8(%r12){1to2}, %xmm1, %xmm2                   #402.12
        vpermilpd $1, %xmm2, %xmm3                              #403.12
        addq      $-96, %rbp                                    #382.31
        vfmaddsub231pd (%r12){1to2}, %xmm1, %xmm3               #404.12
        decq      %r14                                          #382.31
        vsubpd    %xmm3, %xmm0, %xmm1                           #405.12
        vmovupd   %xmm1, (%r9)                                  #406.22
        addq      $-96, %r9                                     #382.31
        addq      $-16, %r12                                    #382.31
        decl      %r13d                                         #382.31
        jns       ..B3.4        # Prob 82%                      #382.28
                                # LOE rax rdx rbx rbp rsi rdi r8 r9 r11 r12 r14 r13d xmm0
..B3.10:                        # Preds ..B3.8 ..B3.2
                                # Execution count [5.00e+00]
        .byte     15                                            #381.22
        .byte     31                                            #381.22
        .byte     128                                           #381.22
        .byte     0                                             #381.22
        .byte     0                                             #381.22
        .byte     0                                             #381.22
        .byte     0                                             #381.22
        addq      $-16, %rax                                    #381.22
        decq      %rdx                                          #381.22
        jg        ..B3.2        # Prob 80%                      #381.19
                                # LOE rax rdx rsi rdi xmm0
..B3.11:                        # Preds ..B3.10
                                # Execution count [1.00e+00]
	.cfi_restore 6
        popq      %rbp                                          #409.1
	.cfi_def_cfa_offset 48
	.cfi_restore 3
        popq      %rbx                                          #409.1
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #409.1
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #409.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #409.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #409.1
	.cfi_def_cfa_offset 8
        ret                                                     #409.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	solv_sys_avx512,@function
	.size	solv_sys_avx512,.-solv_sys_avx512
	.data
# -- End  solv_sys_avx512
	.text
# -- Begin  bck_house_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl bck_house_avx512
# --- bck_house_avx512(complex_dble *, complex_dble *, double *)
bck_house_avx512:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
..B4.1:                         # Preds ..B4.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_bck_house_avx512.62:
..L63:
                                                         #412.1
        pushq     %r12                                          #412.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #412.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #412.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #412.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rbx                                          #412.1
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
        pushq     %rbp                                          #412.1
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
        movq      %rsi, %r8                                     #412.1
        movq      %rdx, %r9                                     #412.1
        xorl      %edx, %edx                                    #419.3
        xorl      %esi, %esi                                    #419.3
        vxorpd    %xmm0, %xmm0, %xmm0                           #436.12
        movq      80(%r8), %rax                                 #416.15
        movq      88(%r8), %rcx                                 #417.15
        movq      %rax, 560(%rdi)                               #416.3
        xorl      %eax, %eax                                    #431.26
        movq      %rcx, 568(%rdi)                               #417.3
        xorl      %ecx, %ecx                                    #419.3
                                # LOE rax rdx rdi r8 r9 ecx esi xmm0
..B4.2:                         # Preds ..B4.15 ..B4.1
                                # Execution count [5.00e+00]
        movl      %edx, %r12d                                   #420.12
        movq      %r8, %r11                                     #420.12
        movq      %r12, %rbp                                    #420.12
        movslq    %esi, %r15                                    #422.16
        shlq      $4, %rbp                                      #420.12
        shlq      $4, %r15                                      #422.16
        subq      %rbp, %r11                                    #420.12
        movq      448(%rdi,%r15), %r13                          #422.16
        movq      64(%r11), %rbx                                #420.12
        movq      %r13, 64(%r11)                                #422.5
        lea       1(%rdx), %r13d                                #427.5
        movq      %rbx, 448(%rdi,%r15)                          #424.5
        lea       5(%rcx), %ebx                                 #427.10
        movq      456(%rdi,%r15), %r10                          #423.16
        movq      72(%r11), %r14                                #421.12
        movq      %r10, 72(%r11)                                #423.5
        movq      %r14, 456(%rdi,%r15)                          #425.5
        cmpl      $6, %ebx                                      #427.27
        jge       ..B4.9        # Prob 50%                      #427.27
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx esi r13d xmm0
..B4.3:                         # Preds ..B4.2
                                # Execution count [5.00e+00]
        xorl      %r14d, %r14d                                  #427.5
        lea       1(%rdx), %r15d                                #427.5
        shrl      $1, %r15d                                     #427.5
        movl      $1, %ebx                                      #427.5
        xorl      %r10d, %r10d                                  #428.7
        testl     %r15d, %r15d                                  #427.5
        jbe       ..B4.7        # Prob 9%                       #427.5
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx ebx esi r10d r13d r14d r15d xmm0
..B4.4:                         # Preds ..B4.3
                                # Execution count [4.50e+00]
        movq      %r8, -24(%rsp)                                #[spill]
        movq      %r9, -16(%rsp)                                #[spill]
        .align    16,0x90
                                # LOE rax rdx rbp rdi r11 r12 ecx esi r10d r13d r14d r15d xmm0
..B4.5:                         # Preds ..B4.5 ..B4.4
                                # Execution count [1.25e+01]
        lea       (%rsi,%r10), %ebx                             #429.18
        addl      $12, %r10d                                    #427.5
        movslq    %ebx, %rbx                                    #429.18
        lea       (%r14,%r14), %r8d                             #429.7
        movslq    %r8d, %r8                                     #429.7
        incl      %r14d                                         #427.5
        shlq      $4, %rbx                                      #429.18
        shlq      $4, %r8                                       #429.7
        movq      552(%rdi,%rbx), %r9                           #429.18
        movq      %r9, 88(%r11,%r8)                             #429.7
        movq      544(%rdi,%rbx), %r9                           #428.18
        movq      %r9, 80(%r11,%r8)                             #428.7
        movq      648(%rdi,%rbx), %r9                           #429.18
        movq      %rax, 552(%rdi,%rbx)                          #431.7
        movq      %r9, 104(%r11,%r8)                            #429.7
        movq      640(%rdi,%rbx), %r9                           #428.18
        movq      %rax, 544(%rdi,%rbx)                          #430.7
        movq      %r9, 96(%r11,%r8)                             #428.7
        movq      %rax, 648(%rdi,%rbx)                          #431.7
        movq      %rax, 640(%rdi,%rbx)                          #430.7
        cmpl      %r15d, %r14d                                  #427.5
        jb        ..B4.5        # Prob 63%                      #427.5
                                # LOE rax rdx rbp rdi r11 r12 ecx esi r10d r13d r14d r15d xmm0
..B4.6:                         # Preds ..B4.5
                                # Execution count [4.50e+00]
        movq      -24(%rsp), %r8                                #[spill]
        lea       1(%r14,%r14), %ebx                            #428.7
        movq      -16(%rsp), %r9                                #[spill]
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx ebx esi r13d xmm0
..B4.7:                         # Preds ..B4.6 ..B4.3
                                # Execution count [5.00e+00]
        lea       -1(%rbx), %r10d                               #427.5
        cmpl      %r13d, %r10d                                  #427.5
        jae       ..B4.9        # Prob 9%                       #427.5
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx ebx esi r13d xmm0
..B4.8:                         # Preds ..B4.7
                                # Execution count [4.50e+00]
        movslq    %ebx, %r10                                    #429.7
        lea       (%rbx,%rbx,2), %ebx                           #429.18
        subq      %r12, %r10                                    #429.7
        lea       (%rsi,%rbx,2), %r14d                          #429.18
        movslq    %r14d, %r14                                   #429.18
        shlq      $4, %r14                                      #429.18
        shlq      $4, %r10                                      #429.7
        movq      456(%rdi,%r14), %r15                          #429.18
        movq      %r15, 72(%r8,%r10)                            #429.7
        movq      448(%rdi,%r14), %r15                          #428.18
        movq      %r15, 64(%r8,%r10)                            #428.7
        movq      %rax, 456(%rdi,%r14)                          #431.7
        movq      %rax, 448(%rdi,%r14)                          #430.7
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx esi r13d xmm0
..B4.9:                         # Preds ..B4.2 ..B4.8 ..B4.7
                                # Execution count [3.96e-01]
        shlq      $3, %r12                                      #448.28
        negq      %rbp                                          #439.30
        negq      %r12                                          #448.28
        addq      %r9, %r12                                     #448.28
        addq      %rdi, %rbp                                    #439.30
        addq      $2, %rdx                                      #438.23
        xorb      %bl, %bl                                      #434.5
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx esi r13d bl xmm0
..B4.10:                        # Preds ..B4.14 ..B4.9
                                # Execution count [2.54e+01]
        movq      %rax, %r14                                    #438.7
        vmovapd   %xmm0, %xmm1                                  #436.12
        movq      %r14, %r10                                    #438.7
                                # LOE rax rdx rbp rdi r8 r9 r10 r11 r12 r14 ecx esi r13d bl xmm0 xmm1
..B4.11:                        # Preds ..B4.11 ..B4.10
                                # Execution count [1.41e+02]
        vmovupd   64(%r10,%r11), %xmm3                          #441.27
        incq      %r14                                          #438.7
        vmulpd    72(%r10,%rbp){1to2}, %xmm3, %xmm2             #442.14
        vpermilpd $1, %xmm2, %xmm4                              #443.14
        vfmaddsub231pd 64(%r10,%rbp){1to2}, %xmm3, %xmm4        #444.14
        addq      $16, %r10                                     #438.7
        vaddpd    %xmm1, %xmm4, %xmm1                           #445.14
        cmpq      %rdx, %r14                                    #438.7
        jb        ..B4.11       # Prob 82%                      #438.7
                                # LOE rax rdx rbp rdi r8 r9 r10 r11 r12 r14 ecx esi r13d bl xmm0 xmm1
..B4.12:                        # Preds ..B4.11
                                # Execution count [2.70e+01]
        movq      %rax, %r15                                    #451.7
        lea       64(%rbp), %r10                                #451.7
        vmulpd    32(%r12){1to2}, %xmm1, %xmm1                  #449.12
        movq      %r15, %r14                                    #451.7
                                # LOE rax rdx rbp rdi r8 r9 r10 r11 r12 r14 r15 ecx esi r13d bl xmm0 xmm1
..B4.13:                        # Preds ..B4.13 ..B4.12
                                # Execution count [1.50e+02]
        vmulpd    72(%r14,%r11){1to2}, %xmm1, %xmm2             #454.14
        vpermilpd $1, %xmm2, %xmm4                              #455.14
        incq      %r15                                          #451.7
        vfmsubadd231pd 64(%r14,%r11){1to2}, %xmm1, %xmm4        #456.14
        addq      $16, %r14                                     #451.7
        vmovupd   (%r10), %xmm3                                 #458.28
        vsubpd    %xmm4, %xmm3, %xmm5                           #459.14
        vmovupd   %xmm5, (%r10)                                 #460.24
        addq      $16, %r10                                     #451.7
        cmpq      %rdx, %r15                                    #451.7
        jb        ..B4.13       # Prob 82%                      #451.7
                                # LOE rax rdx rbp rdi r8 r9 r10 r11 r12 r14 r15 ecx esi r13d bl xmm0 xmm1
..B4.14:                        # Preds ..B4.13
                                # Execution count [3.00e+01]
        incb      %bl                                           #434.5
        addq      $96, %rbp                                     #434.5
        cmpb      $6, %bl                                       #434.5
        jb        ..B4.10       # Prob 83%                      #434.5
                                # LOE rax rdx rbp rdi r8 r9 r11 r12 ecx esi r13d bl xmm0
..B4.15:                        # Preds ..B4.14
                                # Execution count [5.00e+00]
        addl      $-7, %esi                                     #427.5
        decl      %ecx                                          #427.5
        movl      %r13d, %edx                                   #419.3
        cmpl      $5, %r13d                                     #419.3
        jb        ..B4.2        # Prob 79%                      #419.3
                                # LOE rax rdx rdi r8 r9 ecx esi xmm0
..B4.16:                        # Preds ..B4.15
                                # Execution count [1.00e+00]
	.cfi_restore 6
        popq      %rbp                                          #464.1
	.cfi_def_cfa_offset 48
	.cfi_restore 3
        popq      %rbx                                          #464.1
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #464.1
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #464.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #464.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #464.1
	.cfi_def_cfa_offset 8
        ret                                                     #464.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	bck_house_avx512,@function
	.size	bck_house_avx512,.-bck_house_avx512
	.data
# -- End  bck_house_avx512
	.section .rodata, "a"
	.space 8, 0x00 	# pad
	.align 64
.L_2il0floatpacket.9:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000
	.type	.L_2il0floatpacket.9,@object
	.size	.L_2il0floatpacket.9,64
	.align 64
.L_2il0floatpacket.10:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.10,@object
	.size	.L_2il0floatpacket.10,64
	.align 64
.L_2il0floatpacket.11:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000
	.type	.L_2il0floatpacket.11,@object
	.size	.L_2il0floatpacket.11,64
	.align 64
.L_2il0floatpacket.12:
	.long	0x00000006,0x00000000,0x00000007,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.12,@object
	.size	.L_2il0floatpacket.12,64
	.align 64
.L_2il0floatpacket.13:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.13,@object
	.size	.L_2il0floatpacket.13,64
	.align 64
.L_2il0floatpacket.14:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.14,@object
	.size	.L_2il0floatpacket.14,64
	.align 64
.L_2il0floatpacket.15:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000002,0x00000000,0x00000002,0x00000000,0x0000000c,0x00000000,0x0000000c,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.15,@object
	.size	.L_2il0floatpacket.15,64
	.align 64
.L_2il0floatpacket.16:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x0000000d,0x00000000,0x0000000d,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.16,@object
	.size	.L_2il0floatpacket.16,64
	.align 64
.L_2il0floatpacket.17:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000004,0x00000000,0x00000005,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000
	.type	.L_2il0floatpacket.17,@object
	.size	.L_2il0floatpacket.17,64
	.align 64
.L_2il0floatpacket.18:
	.long	0x00000004,0x00000000,0x00000004,0x00000000,0x00000006,0x00000000,0x00000006,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000,0x0000000b,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.18,@object
	.size	.L_2il0floatpacket.18,64
	.align 64
.L_2il0floatpacket.19:
	.long	0x00000005,0x00000000,0x00000005,0x00000000,0x00000007,0x00000000,0x00000007,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000
	.type	.L_2il0floatpacket.19,@object
	.size	.L_2il0floatpacket.19,64
	.align 64
.L_2il0floatpacket.20:
	.long	0x00000001,0x00000000,0x00000001,0x00000000,0x00000003,0x00000000,0x00000003,0x00000000,0x0000000c,0x00000000,0x0000000c,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.20,@object
	.size	.L_2il0floatpacket.20,64
	.align 64
.L_2il0floatpacket.21:
	.long	0x00000004,0x00000000,0x00000004,0x00000000,0x00000006,0x00000000,0x00000006,0x00000000,0x0000000c,0x00000000,0x0000000c,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.21,@object
	.size	.L_2il0floatpacket.21,64
	.align 64
.L_2il0floatpacket.22:
	.long	0x00000004,0x00000000,0x00000005,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000,0x00000006,0x00000000,0x00000007,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.22,@object
	.size	.L_2il0floatpacket.22,64
	.align 64
.L_2il0floatpacket.23:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000002,0x00000000,0x00000002,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x0000000a,0x00000000,0x0000000a,0x00000000
	.type	.L_2il0floatpacket.23,@object
	.size	.L_2il0floatpacket.23,64
	.align 64
.L_2il0floatpacket.24:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000,0x0000000b,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.24,@object
	.size	.L_2il0floatpacket.24,64
	.align 64
.L_2il0floatpacket.25:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000002,0x00000000,0x00000002,0x00000000,0x0000000d,0x00000000,0x0000000d,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.25,@object
	.size	.L_2il0floatpacket.25,64
	.align 64
.L_2il0floatpacket.26:
	.long	0x00000001,0x00000000,0x00000001,0x00000000,0x00000003,0x00000000,0x00000003,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000
	.type	.L_2il0floatpacket.26,@object
	.size	.L_2il0floatpacket.26,64
	.align 64
.L_2il0floatpacket.27:
	.long	0x00000004,0x00000000,0x00000004,0x00000000,0x00000006,0x00000000,0x00000006,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x0000000a,0x00000000,0x0000000a,0x00000000
	.type	.L_2il0floatpacket.27,@object
	.size	.L_2il0floatpacket.27,64
	.align 64
.L_2il0floatpacket.28:
	.long	0x00000005,0x00000000,0x00000005,0x00000000,0x00000007,0x00000000,0x00000007,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000,0x0000000b,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.28,@object
	.size	.L_2il0floatpacket.28,64
	.align 64
.L_2il0floatpacket.29:
	.long	0x00000001,0x00000000,0x00000001,0x00000000,0x00000003,0x00000000,0x00000003,0x00000000,0x0000000d,0x00000000,0x0000000d,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.29,@object
	.size	.L_2il0floatpacket.29,64
	.align 64
.L_2il0floatpacket.30:
	.long	0x00000000,0x00000000,0x00000000,0x00000000,0x00000008,0x00000000,0x00000008,0x00000000,0x00000006,0x00000000,0x00000006,0x00000000,0x0000000e,0x00000000,0x0000000e,0x00000000
	.type	.L_2il0floatpacket.30,@object
	.size	.L_2il0floatpacket.30,64
	.align 64
.L_2il0floatpacket.31:
	.long	0x00000001,0x00000000,0x00000001,0x00000000,0x00000009,0x00000000,0x00000009,0x00000000,0x00000007,0x00000000,0x00000007,0x00000000,0x0000000f,0x00000000,0x0000000f,0x00000000
	.type	.L_2il0floatpacket.31,@object
	.size	.L_2il0floatpacket.31,64
	.align 64
.L_2il0floatpacket.32:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000
	.type	.L_2il0floatpacket.32,@object
	.size	.L_2il0floatpacket.32,64
	.align 64
.L_2il0floatpacket.33:
	.long	0x00000000,0x00000000,0x00000001,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000,0x00000002,0x00000000,0x00000003,0x00000000,0x0000000a,0x00000000,0x0000000b,0x00000000
	.type	.L_2il0floatpacket.33,@object
	.size	.L_2il0floatpacket.33,64
	.align 64
.L_2il0floatpacket.34:
	.long	0x00000006,0x00000000,0x00000007,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x00000008,0x00000000,0x00000009,0x00000000
	.type	.L_2il0floatpacket.34,@object
	.size	.L_2il0floatpacket.34,64
	.align 64
.L_2il0floatpacket.35:
	.long	0x00000002,0x00000000,0x00000003,0x00000000,0x0000000e,0x00000000,0x0000000f,0x00000000,0x00000000,0x00000000,0x00000001,0x00000000,0x0000000c,0x00000000,0x0000000d,0x00000000
	.type	.L_2il0floatpacket.35,@object
	.size	.L_2il0floatpacket.35,64
	.align 16
.L_2il0floatpacket.37:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.37,@object
	.size	.L_2il0floatpacket.37,16
	.align 8
.L_2il0floatpacket.36:
	.long	0x00000000,0x3cb00000
	.type	.L_2il0floatpacket.36,@object
	.size	.L_2il0floatpacket.36,8
	.align 8
.L_2il0floatpacket.38:
	.long	0x00000000,0x3ff00000
	.type	.L_2il0floatpacket.38,@object
	.size	.L_2il0floatpacket.38,8
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
