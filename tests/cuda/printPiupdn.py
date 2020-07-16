SIZE = 8


def binary_to_decimal(binary):
    '''
    Convert a binary little-endian array to 32bit integers
    '''
    values = []
    for i in range(2 * SIZE**4):
        start = 4*i
        end = 4*i + 4
        value = binary[start:end]
        values.append(int.from_bytes(value, byteorder='little'))
    return values


f = open('./' + str(SIZE) + '/dataBefore/piup.bin', 'rb')
piupBin = f.read()
f.close()
piup = binary_to_decimal(piupBin)

f = open('./' + str(SIZE) + '/dataBefore/pidn.bin', 'rb')
piupBin = f.read()
f.close()
pidn = binary_to_decimal(piupBin)

print("          piup            --         pidn")
print("====================================================")
for i in range(SIZE**4//2):
    upval1 = piup[4*i + 0]
    upval2 = piup[4*i + 1]
    upval3 = piup[4*i + 2]
    upval4 = piup[4*i + 3]
    dnval1 = pidn[4*i + 0]
    dnval2 = pidn[4*i + 1]
    dnval3 = pidn[4*i + 2]
    dnval4 = pidn[4*i + 3]
    print("{0:5d} {1:5d} {2:5d} {3:5d}   -- {4:5d} {5:5d} {6:5d} {7:5d}".format(upval1, upval2, upval3, upval4, dnval1, dnval2, dnval3, dnval4))
