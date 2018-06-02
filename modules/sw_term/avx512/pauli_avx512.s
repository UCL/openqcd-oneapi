# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 17.0.4.196 Build 20170411";
# mark_description "-I../../..//include -I.. -I/cineca/prod/opt/compilers/intel/pe-xe-2017/binary/impi/2017.3.196/intel64/includ";
# mark_description "e -isystem /cineca/prod/opt/compilers/intel/pe-xe-2018/binary/impi/2018.1.163/include64/ -DNPROC0=1 -DNPROC1";
# mark_description "=1 -DNPROC2=1 -DNPROC3=1 -DL0=8 -DL1=8 -DL2=8 -DL3=8 -DNPROC0_BLK=1 -DNPROC1_BLK=1 -DNPROC2_BLK=1 -DNPROC3_B";
# mark_description "LK=1 -std=c89 -xCORE-AVX512 -mtune=skylake -DAVX512 -O3 -Ddirac_counters -pedantic -fstrict-aliasing -Wno-lo";
# mark_description "ng-long -Wstrict-prototypes -S";
	.file "pauli_avx512.c"
	.text
..TXTST0:
# -- Begin  mul_pauli2_avx512
	.text
# mark_begin;
       .align    16,0x90
	.globl mul_pauli2_avx512
# --- mul_pauli2_avx512(float, const pauli *, const spinor *, spinor *)
mul_pauli2_avx512:
# parameter 1: %xmm0
# parameter 2: %rdi
# parameter 3: %rsi
# parameter 4: %rdx
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_mul_pauli2_avx512.1:
..L2:
                                                          #12.1
        movl      $42410, %eax                                  #79.9
        vmovups   (%rdi), %zmm8                                 #48.27
        vmovups   64(%rdi), %zmm7                               #49.27
        vmovups   144(%rdi), %zmm29                             #51.27
        vmovups   208(%rdi), %zmm3                              #52.27
        vmovups   .L_2il0floatpacket.11(%rip), %zmm25           #66.9
        vmovups   .L_2il0floatpacket.12(%rip), %zmm31           #68.9
        vmovups   .L_2il0floatpacket.13(%rip), %zmm16           #71.9
        vbroadcastss %xmm0, %xmm14                              #61.9
        vmovups   64(%rsi), %ymm12                              #40.9
        vmovups   (%rsi), %zmm11                                #39.29
        vmovups   .L_2il0floatpacket.8(%rip), %zmm9             #42.9
        vmovups   .L_2il0floatpacket.14(%rip), %zmm17           #77.10
        vmovups   .L_2il0floatpacket.15(%rip), %zmm19           #85.10
        vmovups   .L_2il0floatpacket.17(%rip), %zmm22           #98.9
        vmovups   .L_2il0floatpacket.9(%rip), %zmm10            #44.9
        vmovups   .L_2il0floatpacket.10(%rip), %zmm13           #46.9
        vmovups   .L_2il0floatpacket.16(%rip), %zmm20           #90.10
        vmovups   .L_2il0floatpacket.18(%rip), %zmm24           #104.10
        vmovups   .L_2il0floatpacket.19(%rip), %zmm28           #113.9
        vmovups   128(%rdi), %zmm2                              #50.27
        vpermi2ps %zmm29, %zmm8, %zmm25                         #66.9
        vpermi2ps %zmm3, %zmm7, %zmm31                          #68.9
        vbroadcastss %xmm14, %zmm26                             #62.9
        vpermt2ps %zmm29, %zmm28, %zmm8                         #113.9
        vpermi2ps %zmm31, %zmm25, %zmm16                        #71.9
        vpermi2ps %zmm25, %zmm31, %zmm19                        #85.10
        vpermt2ps %zmm31, %zmm22, %zmm25                        #98.9
        vpermi2ps %zmm16, %zmm26, %zmm17                        #77.10
        vpermi2ps %zmm26, %zmm31, %zmm20                        #90.10
        vpermt2ps %zmm25, %zmm24, %zmm26                        #104.10
        vmovups   .L_2il0floatpacket.22(%rip), %zmm28           #134.9
        vshufps   $170, %zmm31, %zmm8, %zmm30                   #115.10
        vshufps   $255, %zmm31, %zmm8, %zmm24                   #118.10
        kmovw     %eax, %k2                                     #79.9
        vpermilps $160, %zmm16, %zmm15                          #72.10
        movl      $23125, %eax                                  #80.9
        kmovw     %eax, %k3                                     #80.9
        movl      $21925, %eax                                  #92.9
        kmovw     %eax, %k4                                     #92.9
        vpermilps $160, %zmm25, %zmm23                          #99.10
        movl      $43610, %eax                                  #93.9
        kmovw     %eax, %k5                                     #93.9
        movl      $26022, %eax                                  #106.9
        kmovw     %eax, %k6                                     #106.9
        movl      $39513, %eax                                  #107.9
        kmovw     %eax, %k7                                     #107.9
        movl      $43690, %eax                                  #120.9
        vpermi2ps %zmm12, %zmm11, %zmm9                         #42.9
        vpermi2ps %zmm12, %zmm11, %zmm10                        #44.9
        vpermt2ps %zmm12, %zmm13, %zmm11                        #46.9
        vmulps    %zmm15, %zmm9, %zmm1                          #73.9
        vmulps    %zmm19, %zmm10, %zmm0                         #86.9
        vmulps    %zmm23, %zmm11, %zmm12                        #100.9
        vmovups   .L_2il0floatpacket.21(%rip), %zmm15           #130.9
        vpermilps $177, %zmm9, %zmm5                            #56.10
        vmulps    %zmm5, %zmm17, %zmm18                         #78.10
        vmovups   .L_2il0floatpacket.23(%rip), %zmm17           #155.9
        vpermi2ps %zmm3, %zmm7, %zmm15                          #130.9
        vaddps    %zmm18, %zmm1, %zmm1{%k2}                     #79.9
        vpermi2ps %zmm15, %zmm8, %zmm28                         #134.9
        vsubps    %zmm18, %zmm1, %zmm1{%k3}                     #80.9
        vpermi2ps %zmm15, %zmm8, %zmm17                         #155.9
        kmovw     %eax, %k3                                     #120.9
        vfmadd231ps %zmm10, %zmm30, %zmm1                       #116.9
        vpermilps $177, %zmm11, %zmm4                           #58.10
        movl      $21845, %eax                                  #121.9
        vmulps    %zmm4, %zmm26, %zmm27                         #105.10
        vmovups   .L_2il0floatpacket.20(%rip), %zmm26           #128.10
        kmovw     %eax, %k2                                     #121.9
        vpermt2ps 272(%rdi), %zmm26, %zmm2                      #128.10
        vaddps    %zmm27, %zmm12, %zmm12{%k6}                   #106.9
        vpermilps $177, %zmm10, %zmm6                           #57.10
        movl      $61680, %eax                                  #136.10
        vmulps    %zmm6, %zmm20, %zmm21                         #91.10
        vmulps    %zmm24, %zmm6, %zmm25                         #119.10
        vmovups   .L_2il0floatpacket.24(%rip), %zmm20           #166.9
        vsubps    %zmm27, %zmm12, %zmm12{%k7}                   #107.9
        vaddps    %zmm21, %zmm0, %zmm0{%k4}                     #92.9
        vpermt2ps %zmm3, %zmm20, %zmm7                          #166.9
        vaddps    %zmm25, %zmm1, %zmm1{%k3}                     #120.9
        vsubps    %zmm21, %zmm0, %zmm0{%k5}                     #93.9
        vmovups   .L_2il0floatpacket.25(%rip), %zmm3            #169.9
        vmovups   .L_2il0floatpacket.26(%rip), %zmm21           #192.9
        vsubps    %zmm25, %zmm1, %zmm1{%k2}                     #121.9
        vpermi2ps %zmm8, %zmm7, %zmm3                           #169.9
        kmovw     %eax, %k1                                     #136.10
        vpermilps $245, %zmm28, %zmm29                          #139.9
        movl      $3855, %eax                                   #145.9
        vshufps   $244, %zmm2, %zmm29, %zmm29{%k1}              #140.10
        vshufps   $68, %zmm2, %zmm7, %zmm7{%k1}                 #180.9
        kmovw     %eax, %k4                                     #145.9
        vmulps    %zmm29, %zmm5, %zmm30                         #141.10
        vpermilps $160, %zmm28, %zmm27                          #135.9
        movl      $42405, %eax                                  #151.9
        vshufps   $164, %zmm2, %zmm27, %zmm27{%k1}              #136.10
        vmovaps   %zmm15, %zmm13                                #145.9
        vshufps   $228, %zmm15, %zmm8, %zmm13{%k4}              #145.9
        vfmadd231ps %zmm9, %zmm27, %zmm0                        #137.9
        vpermilps $245, %zmm13, %zmm14                          #149.10
        vmulps    %zmm14, %zmm5, %zmm5                          #150.10
        vaddps    %zmm30, %zmm0, %zmm0{%k2}                     #142.9
        kmovw     %eax, %k2                                     #151.9
        vsubps    %zmm30, %zmm0, %zmm0{%k3}                     #143.9
        vpermilps $160, %zmm13, %zmm31                          #146.10
        movl      $23130, %eax                                  #152.9
        vpermilps $160, %zmm3, %zmm8                            #170.10
        vfmadd213ps %zmm12, %zmm31, %zmm9                       #147.9
        kmovw     %eax, %k3                                     #152.9
        vaddps    %zmm5, %zmm9, %zmm9{%k2}                      #151.9
        vpermilps $160, %zmm17, %zmm16                          #156.10
        movl      $42662, %eax                                  #161.9
        vpermilps $10, %zmm2, %zmm8{%k1}                        #171.10
        vfmadd231ps %zmm11, %zmm16, %zmm1                       #157.9
        vfmadd213ps %zmm0, %zmm8, %zmm11                        #172.9
        vsubps    %zmm5, %zmm9, %zmm9{%k3}                      #152.9
        kmovw     %eax, %k5                                     #161.9
        vpermilps $245, %zmm3, %zmm0                            #174.10
        movl      $22873, %eax                                  #162.9
        vpermilps $245, %zmm17, %zmm18                          #159.10
        vpermilps $95, %zmm2, %zmm0{%k1}                        #175.10
        vpermilps $160, %zmm7, %zmm2                            #181.10
        vpermilps $245, %zmm7, %zmm7                            #184.10
        vmulps    %zmm18, %zmm4, %zmm19                         #160.10
        vmulps    %zmm7, %zmm6, %zmm6                           #185.10
        vmulps    %zmm0, %zmm4, %zmm4                           #176.10
        vfmadd213ps %zmm9, %zmm2, %zmm10                        #182.9
        vmovups   .L_2il0floatpacket.27(%rip), %zmm9            #194.9
        vaddps    %zmm19, %zmm1, %zmm1{%k5}                     #161.9
        vaddps    %zmm6, %zmm10, %zmm10{%k2}                    #186.9
        kmovw     %eax, %k6                                     #162.9
        vsubps    %zmm6, %zmm10, %zmm10{%k3}                    #187.9
        vsubps    %zmm19, %zmm1, %zmm1{%k6}                     #162.9
        movl      $25957, %eax                                  #177.9
        kmovw     %eax, %k7                                     #177.9
        movl      $39578, %eax                                  #178.9
        kmovw     %eax, %k5                                     #178.9
        vaddps    %zmm4, %zmm11, %zmm11{%k7}                    #177.9
        vpermi2ps %zmm10, %zmm1, %zmm21                         #192.9
        vpermt2ps %zmm10, %zmm9, %zmm1                          #194.9
        vsubps    %zmm4, %zmm11, %zmm11{%k5}                    #178.9
        vaddps    %zmm1, %zmm21, %zmm1                          #195.9
        vextractf32x4 $1, %zmm11, %xmm10                        #201.11
        vextractf32x4 $2, %zmm11, %xmm0                         #208.11
        vextractf32x4 $3, %zmm11, %xmm23                        #209.11
        vaddps    %xmm11, %xmm10, %xmm22                        #202.11
        vmovups   %xmm22, 32(%rdx)                              #203.21
        vmovups   %ymm1, (%rdx)                                 #198.24
        vextractf64x4 $1, %zmm1, 48(%rdx)                       #206.24
        vaddps    %xmm0, %xmm23, %xmm1                          #210.11
        vmovups   %xmm1, 80(%rdx)                               #211.21
        vzeroupper                                              #212.1
        ret                                                     #212.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	mul_pauli2_avx512,@function
	.size	mul_pauli2_avx512,.-mul_pauli2_avx512
	.data
# -- End  mul_pauli2_avx512
	.section .rodata, "a"
	.align 64
	.align 64
.L_2il0floatpacket.8:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000006,0x00000007,0x00000008,0x00000009,0x0000000c,0x0000000d,0x0000000e,0x0000000f,0x00000012,0x00000013,0x00000014,0x00000015
	.type	.L_2il0floatpacket.8,@object
	.size	.L_2il0floatpacket.8,64
	.align 64
.L_2il0floatpacket.9:
	.long	0x00000002,0x00000003,0x00000004,0x00000005,0x00000008,0x00000009,0x0000000a,0x0000000b,0x0000000e,0x0000000f,0x00000010,0x00000011,0x00000014,0x00000015,0x00000016,0x00000017
	.type	.L_2il0floatpacket.9,@object
	.size	.L_2il0floatpacket.9,64
	.align 64
.L_2il0floatpacket.10:
	.long	0x00000004,0x00000005,0x00000000,0x00000001,0x0000000a,0x0000000b,0x00000006,0x00000007,0x00000010,0x00000011,0x0000000c,0x0000000d,0x00000016,0x00000017,0x00000012,0x00000013
	.type	.L_2il0floatpacket.10,@object
	.size	.L_2il0floatpacket.10,64
	.align 64
.L_2il0floatpacket.11:
	.long	0x00000000,0x00000001,0x0000000a,0x0000000b,0x00000004,0x00000005,0x00000002,0x00000003,0x00000010,0x00000011,0x0000001a,0x0000001b,0x00000014,0x00000015,0x00000012,0x00000013
	.type	.L_2il0floatpacket.11,@object
	.size	.L_2il0floatpacket.11,64
	.align 64
.L_2il0floatpacket.12:
	.long	0x00000004,0x00000005,0x00000000,0x00000001,0x0000000c,0x0000000d,0x00000006,0x00000007,0x00000014,0x00000015,0x00000010,0x00000011,0x0000001c,0x0000001d,0x00000016,0x00000017
	.type	.L_2il0floatpacket.12,@object
	.size	.L_2il0floatpacket.12,64
	.align 64
.L_2il0floatpacket.13:
	.long	0x00000000,0x00000000,0x00000001,0x00000001,0x00000002,0x00000003,0x00000010,0x00000011,0x00000008,0x00000008,0x00000009,0x00000009,0x0000000a,0x0000000b,0x00000018,0x00000019
	.type	.L_2il0floatpacket.13,@object
	.size	.L_2il0floatpacket.13,64
	.align 64
.L_2il0floatpacket.14:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000015,0x00000015,0x00000017,0x00000017,0x00000008,0x00000009,0x0000000a,0x0000000b,0x0000001d,0x0000001d,0x0000001f,0x0000001f
	.type	.L_2il0floatpacket.14,@object
	.size	.L_2il0floatpacket.14,64
	.align 64
.L_2il0floatpacket.15:
	.long	0x00000000,0x00000000,0x00000004,0x00000004,0x00000014,0x00000014,0x00000015,0x00000015,0x00000008,0x00000008,0x0000000c,0x0000000c,0x0000001c,0x0000001c,0x0000001d,0x0000001d
	.type	.L_2il0floatpacket.15,@object
	.size	.L_2il0floatpacket.15,64
	.align 64
.L_2il0floatpacket.16:
	.long	0x00000001,0x00000001,0x00000005,0x00000005,0x00000014,0x00000015,0x00000016,0x00000017,0x00000009,0x00000009,0x0000000d,0x0000000d,0x0000001c,0x0000001d,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.16,@object
	.size	.L_2il0floatpacket.16,64
	.align 64
.L_2il0floatpacket.17:
	.long	0x00000006,0x00000006,0x00000002,0x00000003,0x00000014,0x00000015,0x00000007,0x00000007,0x0000000e,0x0000000e,0x0000000a,0x0000000b,0x0000001c,0x0000001d,0x0000000f,0x0000000f
	.type	.L_2il0floatpacket.17,@object
	.size	.L_2il0floatpacket.17,64
	.align 64
.L_2il0floatpacket.18:
	.long	0x00000000,0x00000001,0x00000013,0x00000013,0x00000015,0x00000015,0x00000006,0x00000007,0x00000008,0x00000009,0x0000001b,0x0000001b,0x0000001d,0x0000001d,0x0000000e,0x0000000f
	.type	.L_2il0floatpacket.18,@object
	.size	.L_2il0floatpacket.18,64
	.align 64
.L_2il0floatpacket.19:
	.long	0x00000008,0x00000009,0x00000006,0x00000007,0x0000000e,0x0000000f,0x0000000c,0x0000000d,0x00000018,0x00000019,0x00000016,0x00000017,0x0000001e,0x0000001f,0x0000001c,0x0000001d
	.type	.L_2il0floatpacket.19,@object
	.size	.L_2il0floatpacket.19,64
	.align 64
.L_2il0floatpacket.20:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000000,0x00000001,0x00000002,0x00000003,0x00000010,0x00000011,0x00000012,0x00000013,0x00000010,0x00000011,0x00000012,0x00000013
	.type	.L_2il0floatpacket.20,@object
	.size	.L_2il0floatpacket.20,64
	.align 64
.L_2il0floatpacket.21:
	.long	0x00000006,0x00000007,0x00000002,0x00000003,0x00000008,0x00000009,0x0000000e,0x0000000f,0x00000016,0x00000017,0x00000012,0x00000013,0x00000018,0x00000019,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.21,@object
	.size	.L_2il0floatpacket.21,64
	.align 64
.L_2il0floatpacket.22:
	.long	0x00000006,0x00000007,0x00000010,0x00000011,0x00000016,0x00000017,0x00000000,0x00000000,0x0000000e,0x0000000f,0x00000018,0x00000019,0x0000001e,0x0000001f,0x00000000,0x00000000
	.type	.L_2il0floatpacket.22,@object
	.size	.L_2il0floatpacket.22,64
	.align 64
.L_2il0floatpacket.23:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000004,0x00000005,0x00000012,0x00000013,0x00000008,0x00000009,0x0000000a,0x0000000b,0x0000000c,0x0000000d,0x0000001a,0x0000001b
	.type	.L_2il0floatpacket.23,@object
	.size	.L_2il0floatpacket.23,64
	.align 64
.L_2il0floatpacket.24:
	.long	0x00000000,0x00000001,0x00000008,0x00000009,0x0000000a,0x0000000b,0xffffffff,0xffffffff,0x00000010,0x00000011,0x00000018,0x00000019,0x0000001a,0x0000001b,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.24,@object
	.size	.L_2il0floatpacket.24,64
	.align 64
.L_2il0floatpacket.25:
	.long	0x00000004,0x00000005,0x00000014,0x00000015,0x00000000,0x00000000,0x00000000,0x00000000,0x0000000c,0x0000000d,0x0000001c,0x0000001d,0x00000000,0x00000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.25,@object
	.size	.L_2il0floatpacket.25,64
	.align 64
.L_2il0floatpacket.26:
	.long	0x00000000,0x00000001,0x00000002,0x00000003,0x00000010,0x00000011,0x00000012,0x00000013,0x00000008,0x00000009,0x0000000a,0x0000000b,0x00000018,0x00000019,0x0000001a,0x0000001b
	.type	.L_2il0floatpacket.26,@object
	.size	.L_2il0floatpacket.26,64
	.align 64
.L_2il0floatpacket.27:
	.long	0x00000004,0x00000005,0x00000006,0x00000007,0x00000014,0x00000015,0x00000016,0x00000017,0x0000000c,0x0000000d,0x0000000e,0x0000000f,0x0000001c,0x0000001d,0x0000001e,0x0000001f
	.type	.L_2il0floatpacket.27,@object
	.size	.L_2il0floatpacket.27,64
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
