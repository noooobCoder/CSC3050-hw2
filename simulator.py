import numpy
import os
import sys

checkfile = sys.argv[3]
with open(checkfile, 'r') as ck:
    checkList = ck.readlines()


infile = sys.argv[4]
outfile = sys.argv[5]
inn = open(infile, 'r')
out = open(outfile, 'w')


memoryList = [0] * 1572864
static_data = 262144
pc = 0
hi = 0
lo = 0
Exit = False

funct = {
    '100000': 'add',
    '100001': 'addu',
    '100100': 'and',
    '011010': 'div',
    '011011': 'divu',
    '001001': 'jalr',
    '001000': 'jr',
    '010000': 'mfhi',
    '010010': 'mflo',
    '010001': 'mthi',
    '010011': 'mtlo',
    '011000': 'mult',
    '011001': 'multu',
    '100111': 'nor',
    '100101': 'or',
    '000000': 'sll',
    '000100': 'sllv',
    '101010': 'slt',
    '101011': 'sltu',
    '000011': 'sra',
    '000111': 'srav',
    '000010': 'srl',
    '000110': 'srlv',
    '100010': 'sub',
    '100011': 'subu',
    '001100': 'syscall',
    '100110': 'xor'
}

opCode = {
    '001000': 'addi',
    '001001': 'addiu',
    '001100': 'andi',
    '000100': 'beq',
    '000001xxxxx00001': 'bgez',
    '000111': 'bgtz',
    '000110': 'blez',
    '000001xxxxx00000': 'bltz',
    '000101': 'bne',
    '100000': 'lb',
    '100100': 'lbu',
    '100001': 'lh',
    '100101': 'lhu',
    '001111': 'lui',
    '100011': 'lw',
    '001101': 'ori',
    '101000': 'sb',
    '001010': 'slti',
    '001011': 'sltiu',
    '101001': 'sh',
    '101011': 'sw',
    '001110': 'xori',
    '100010': 'lwl',
    '100110': 'lwr',
    '101010': 'swl',
    '101110': 'swr',
    '000010': 'j',
    '000011': 'jal'
}

regValue = {
    '00000': 0,
    '00001': 0,
    '00010': 0,
    '00011': 0,
    '00100': 0,
    '00101': 0,
    '00110': 0,
    '00111': 0,
    '01000': 0,
    '01001': 0,
    '01010': 0,
    '01011': 0,
    '01100': 0,
    '01101': 0,
    '01110': 0,
    '01111': 0,
    '10000': 0,
    '10001': 0,
    '10010': 0,
    '10011': 0,
    '10100': 0,
    '10101': 0,
    '10110': 0,
    '10111': 0,
    '11000': 0,
    '11001': 0,
    '11010': 0,
    '11011': 0,
    '11100': 5275648,
    '11101': 10485760,
    '11110': 10485760,
    '11111': 0
}

machineCode = open(sys.argv[2], 'r')
lines = machineCode.readlines()
for i in range(len(lines)):
    line = lines[i][:-1]
    memoryList[i] = line
machineCode.close()


def findText(line):
    if len(line) >= 5:
        if line[:5] == '.text':
            return True
    return False


def removeCmts(line):
    flag = False
    for i in range(len(line)):
        if line[i] == '"':
            flag = not flag
        if line[i] == '#' and not flag:
            return line[:i]
    return line


def removeLables(line):
    flag = False
    for i in range(len(line)):
        if line[i] == '"':
            flag *= not flag
        if line[i] == ':' and not flag:
            return line[i + 1:]
    return line


asmFile = open(sys.argv[1], 'r')
lines = asmFile.readlines()
for i in range(len(lines)):
    if findText(lines[i]):
        dataSeg = lines[:i]
        del lines[:i+1]
        break

for i in range(len(dataSeg)):
    if dataSeg[i][:5] == '.data':
        continue
    dataSeg[i] = removeCmts(dataSeg[i])
    if dataSeg[i].split() == []:
        continue
    dataSeg[i] = removeLables(dataSeg[i])
    if dataSeg[i].split() == []:
        continue
    dataSeg[i] = dataSeg[i][:-1]
    dot = dataSeg[i].find('.')
    if dataSeg[i].find('.asciiz') == dot:
        s = dataSeg[i].split('"')[1].replace('\\n', '\n')
        s = s.replace('\\0', '\0')
        while len(s) >= 4:
            memoryList[static_data] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s[:4]), 2)
            static_data += 1
            s = s[4:]
        while len(s) < 4:
            s += '\0'
        memoryList[static_data] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s), 2)
        static_data += 1
    elif dataSeg[i].find('.ascii') == dot:
        s = dataSeg[i].split('"')[1].replace('\\n', '\n')
        s = s.replace('\\0', '\0')
        while len(s) > 4:
            memoryList[static_data] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s[:4]), 2)
            static_data += 1
            s = s[4:]
        while 0 < len(s) < 4:
            s += '\0'
        memoryList[static_data] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s), 2)
        static_data += 1
    elif dataSeg[i].find('.word') == dot:
        dataSeg[i] = dataSeg[i].replace(', ', ',')
        wordList = dataSeg[i].split(' ')[-1].split(',')
        for word in wordList:
            memoryList[static_data] = int(word)
            static_data += 1
    elif dataSeg[i].find('.byte') == dot:
        dataSeg[i] = dataSeg[i].replace(', ', ',')
        byteList = dataSeg[i].split(' ')[-1].split()[0].split(',')
        num = []
        for byte in byteList:
            num.append(int(byte))
            if len(num) == 4:
                memoryList[static_data] = int(''.join(numpy.binary_repr(c, 8)for c in num), 2)
                static_data += 1
                num = []
        while 0 < len(num) < 4:
            num.append(0)
        if num != []:
            memoryList[static_data] = int(''.join(numpy.binary_repr(c, 8)for c in num), 2)
            static_data += 1
    elif dataSeg[i].find('.half') == dot:
        dataSeg[i] = dataSeg[i].replace(', ', ',')
        halfList = dataSeg[i].split(' ')[-1].split()[0].split(',')
        num = []
        for half in halfList:
            num.append(int(half))
            if len(num) == 2:
                memoryList[static_data] = int(''.join(numpy.binary_repr(c, 16)for c in num), 2)
                static_data += 1
                num = []
        while 0 < len(num) < 2:
            num.append(0)
        if num != []:
            memoryList[static_data] = int(''.join(numpy.binary_repr(c, 16)for c in num), 2)
            static_data += 1

static_end = 270336
asmFile.close()


def findType(code):
    op = code[0:6]
    if op == '000000':
        func = code[-6:]
        instru = funct[func]
    else:
        if op == '000001':
            op = op + 'x' * 5 + code[11:16]
        instru = opCode[op]
    return instru


def add(rd, rs, rt):
    rd = rs + rt
    return rd


def addu(rd, rs, rt):
    rd = rs % 4294967296 + rt % 4294967296
    return rd


def andR(rd, rs, rt):
    rd = numpy.bitwise_and(rt, rd)
    return rd


def div(rs, rt):
    lo = rs // rt
    hi = rs % rt
    return lo, hi


def divu(rs, rt):
    lo = rs // rt
    hi == rs % rt
    return lo, hi


def jalr(rd, rs, pc):
    rd = 4194304 + pc * 4
    pc = (rs - 4194304) // 4
    return rd, pc


def jr(rs):
    pc = (rs - 4194304) // 4
    return pc


def mfhi(rd, hi):
    rd = hi
    return rd


def mflo(rd, lo):
    rd = lo
    return rd


def mthi(rs, hi):
    hi = rs
    return hi


def mtlo(rs, lo):
    lo = rs
    return lo


def mult(rs, rt):
    res = rs * rt
    lo = res % 4294967296
    hi = res // 4294967296
    return lo, hi


def multu(rs, rt):
    res = rs * rt
    lo = res % 4294967296
    hi = res // 4294967296
    return lo, hi


def nor(rd, rs, rt):
    not_rd = numpy.bitwise_or(rs, rt)
    rd = numpy.bitwise_not(not_rd)
    return rd


def orR(rd, rs, rt):
    rd = numpy.bitwise_or(rs, rt)
    return rd


def sll(rd, rt, sa):
    rd = numpy.left_shift(rt, sa)
    return rd


def sllv(rd, rt, rs):
    rs = rs % 32
    rd = numpy.left_shift(rt, rs)
    return rd


def slt(rd, rs, rt):
    if(rs < rt):
        rd = 1
    else:
        rd = 0
    return rd


def sltu(rd, rs, rt):
    if(numpy.binary_repr(rs) < numpy.binary_repr(rt)):
        rd = 1
    else:
        rd = 0
    return rd


def sra(rd, rt, sa):
    rd = numpy.right_shift(rt, sa)
    return rd


def srav(rd, rt, rs):
    rs = rs % 32
    rd = numpy.right_shift(rt, rs)
    return rd


def srl(rd, rt, sa):
    rt = rt % 4294967296
    rd = numpy.right_shift(rt, sa)
    return rd


def srlv(rd, rt, rs):
    rt = rt % 4294967296
    rd = numpy.right_shift(rt, rs)
    return rd


def sub(rd, rs, rt):
    rd = rs - rt
    return rd


def subu(rd, rs, rt):
    rd = rs % 4294967296 - rt % 4294967296
    return rd


def xor(rd, rs, rt):
    rd = numpy.bitwise_xor(rs, rt)
    return rd


def addi(rt, rs, imm):
    rt = rs + imm
    return rt


def addiu(rt, rs, imm):
    rt = rs % 4294967296 + imm % 4294967296
    return rt


def andi(rt, rs, imm):
    rt = numpy.bitwise_and(rs, imm)
    return rt


def beq(rs, rt, label_address, pc):
    if(rs == rt):
        pc += label_address
    return pc


def bgez(rs, label_address, pc):
    if(rs >= 0):
        pc += label_address
    return pc


def bgtz(rs, label_address, pc):
    if(rs > 0):
        pc += label_address
    return pc


def blez(rs, label_address, pc):
    if(rs <= 0):
        pc += label_address
    return pc


def bltz(rs, label_address, pc):
    if(rs < 0):
        pc += label_address
    return pc


def bne(rs, rt, label_address, pc):
    if(rs != rt):
        pc += label_address
    return pc


def lb(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[point: point + 8]
    rt = int(temp[0], 2) * (-128) + int(temp[1:], 2)
    return rt


def lbu(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[point:point + 8]
    rt = int(temp, 2)
    return rt


def lh(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[point:point + 16]
    rt = int(temp[0], 2) * (-32768) + int(temp[1:], 2)
    return rt


def lhu(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[point:point + 16]
    rt = int(temp, 2)
    return rt


def lui(rt, imm):
    imm = imm * 65536
    rt = imm
    return rt


def lw(rt, rs, imm):
    rt = memoryList[(rs + imm - 4194304) // 4]
    return rt


def ori(rt, rs, imm):
    rt = numpy.bitwise_or(rs, imm)
    return rt


def sb(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[:point] + numpy.binary_repr(rt, 8) + temp[point + 8:]
    memoryList[(rs + imm - 4194304) // 4] = int(temp, 2)


def slti(rt, rs, imm):
    if(rs < imm):
        rt = 1
    else:
        rt = 0
    return rt


def sltiu(rt, rs, imm):
    if(numpy.binary_repr(rs) < numpy.binary_repr(imm)):
        rt = 1
    else:
        rt = 0
    return rt


def sh(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[:point] + numpy.binary_repr(rt, 16) + temp[point + 16:]
    memoryList[(rs + imm - 4194304) // 4] = int(temp, 2)


def sw(rt, rs, imm):
    memoryList[(rs + imm - 4194304) // 4] = rt


def xori(rt, rs, imm):
    rt = numpy.bitwise_xor(rs, imm)
    return rt


def lwl(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    imm = 3 - imm % 4
    point = ((rs + imm - 4194304) % 4) * 8
    temp = temp[point:]
    rt = numpy.binary_repr(rt, 32)
    rt = temp + rt[32 - point:]
    rt = int(rt, 2)
    return rt


def lwr(rt, rs, imm):
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    imm = 3 - imm % 4
    point = ((rs + imm - 4194304) % 4 + 1) * 8
    temp = temp[:point]
    rt = numpy.binary_repr(rt, 32)
    rt = rt[:32 - point] + temp
    rt = int(rt, 2)
    return rt


def swl(rt, rs, imm):
    rt = numpy.binary_repr(rt, 32)
    imm = 3 - imm % 4
    point = ((rs + imm - 4194304) % 4) * 8
    rt = rt[:32 - point]
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    temp = temp[:point] + rt
    temp = int(temp, 2)
    memoryList[(rs + imm - 4194304) // 4] = temp


def swr(rt, rs, imm):
    rt = numpy.binary_repr(rt, 32)
    imm = 3 - imm % 4
    point = ((rs + imm - 4194304) % 4 + 1) * 8
    rt = rt[32 - point:]
    temp = numpy.binary_repr(memoryList[(rs + imm - 4194304) // 4], 32)
    temp = rt + temp[point:]
    temp = int(temp, 2)
    memoryList[(rs + imm - 4194304) // 4] = temp


def j(address):
    pc = (address - 4194304) // 4
    return pc


def jal(address, pc):
    ra = 4194304 + pc * 4
    pc = (address - 4194304) // 4
    return ra, pc


def syscall(v0, a0, a1, a2):
    global Exit
    global static_end
    if v0 == 1:
        temp = numpy.binary_repr(a0, 32)
        a0 = int(temp[0], 2) * (-2147483648) + int(temp[1:], 2)
        out.write(str(a0))
    if v0 == 4:
        address = (a0 - 4194304) // 4
        string_in_bin = numpy.binary_repr(memoryList[address], 32)
        while string_in_bin[-8:] != '00000000':
            address += 1
            string_in_bin += numpy.binary_repr(memoryList[address], 32)
        ch_string = ''
        while string_in_bin:
            ch_in_bin = string_in_bin[:8]
            if ch_in_bin == '00000000':
                break
            ch_string += chr(int(ch_in_bin, 2))
            string_in_bin = string_in_bin[8:]
        out.write(ch_string)
    if v0 == 5:
        regValue['00010'] = int(inn.readline())
    if v0 == 8:
        s = inn.readline()[:a1][:-1]
        address = (a0 - 4194304) // 4
        while len(s) > 4:
            memoryList[address] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s[:4]), 2)
            address += 1
            s = s[4:]
        while 0 < len(s) < 4:
            s += '\0'
        memoryList[address] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s), 2)
        address += 1
    if v0 == 9:
        regValue['00010'] = static_end * 4 + 4194304
        static_end += a0
    if v0 == 10:
        Exit = True
    if v0 == 11:
        out.write(chr(a0))
    if v0 == 12:
        regValue['00010'] = ord(inn.readline()[0])
    if v0 == 13:
        address = (a0 - 4194304) // 4
        string_in_bin = numpy.binary_repr(memoryList[address], 32)
        while string_in_bin[-8:] != '00000000':
            address += 1
            string_in_bin += numpy.binary_repr(memoryList[address], 32)
        ch_string = ''
        while string_in_bin:
            ch_in_bin = string_in_bin[:8]
            if ch_in_bin == '00000000':
                break
            ch_string += chr(int(ch_in_bin, 2))
            string_in_bin = string_in_bin[8:]
        regValue['00100'] = os.open(ch_string, a1, a2)
    if v0 == 14:
        s = os.read(a0, a2).decode()
        regValue['00100'] = len(s)
        address = (a1 - 4194304) // 4
        while len(s) > 4:
            memoryList[address] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s[:4]), 2)
            address += 1
            s = s[4:]
        while 0 < len(s) < 4:
            s += '\0'
        memoryList[address] = int(''.join(numpy.binary_repr(ord(c), 8)for c in s), 2)
        address += 1
    if v0 == 15:
        address = (a1 - 4194304) // 4
        string_in_bin = numpy.binary_repr(memoryList[address], 32)
        while string_in_bin[-8:] != '00000000':
            address += 1
            string_in_bin += numpy.binary_repr(memoryList[address], 32)
        ch_string = ''
        while string_in_bin:
            ch_in_bin = string_in_bin[:8]
            ch_string += chr(int(ch_in_bin, 2))
            if len(ch_string) == a2:
                break
            if ch_in_bin == '00000000':
                break
            string_in_bin = string_in_bin[8:]
        regValue['00100'] = len(ch_string)
        os.write(a0, ch_string.encode())
    if v0 == 16:
        os.close(a0)
    if v0 == 17:
        Exit = True


def write_bin1(name, pc, lo, hi):
    with open(name, 'wb') as f:
        for value in regValue.values():
            bin_data = int(value).to_bytes(4, 'little', signed=True)
            f.write(bin_data)
        pc = int(pc) * 4 + 4194304
        f.write(pc.to_bytes(4, 'little'))
        f.write(int(lo).to_bytes(4, 'little'))
        f.write(int(hi).to_bytes(4, 'little'))


def write_bin2(name):
    with open(name, 'wb') as f:
        for value in memoryList:
            if isinstance(value, str):
                bin_data = int(value, 2).to_bytes(4, 'little')
                f.write(bin_data)
            else:
                bin_data = int(value).to_bytes(4, 'little', signed=True)
                f.write(bin_data)


time = 0
while Exit is False:
    if str(time) + '\n' in checkList:
        name1 = 'register_' + str(time) + '.bin'
        name2 = 'memory_' + str(time) + '.bin'
        write_bin1(name1, pc, lo, hi)
        write_bin2(name2)
    time += 1
    cur = memoryList[pc]
    pc += 1
    rs = cur[6:11]
    rt = cur[11:16]
    rd = cur[16:21]
    sa = cur[21:26]
    imm = cur[-16:]
    target = cur[-26:] + '00'
    imm = int(imm[0], 2) * (-32768) + int(imm[1:], 2)
    target = int(target, 2)
    instruction = findType(cur)
    if instruction == 'add':
        regValue[rd] = add(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'addu':
        regValue[rd] = addu(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'and':
        regValue[rd] = andR(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'div':
        lo, hi = div(regValue[rs], regValue[rt])
    elif instruction == 'divu':
        lo, hi = divu(regValue[rs], regValue[rt])
    elif instruction == 'jalr':
        regValue[rd], pc = jalr(regValue[rd], regValue[rs], pc)
    elif instruction == 'jr':
        pc = jr(regValue[rs])
    elif instruction == 'mfhi':
        regValue[rd] = mfhi(regValue[rd], hi)
    elif instruction == 'mflo':
        regValue[rd] = mflo(regValue[rd], lo)
    elif instruction == 'mthi':
        hi = mthi(regValue[rs], hi)
    elif instruction == 'mtlo':
        lo = mtlo(regValue[rs], lo)
    elif instruction == 'mult':
        lo, hi = mult(regValue[rs], regValue[rt])
    elif instruction == 'multu':
        lo, hi = multu(regValue[rs], regValue[rt])
    elif instruction == 'nor':
        regValue[rd] = nor(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'or':
        regValue[rd] = orR(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'sll':
        regValue[rd] = sll(regValue[rd], regValue[rt], int(sa, 2))
    elif instruction == 'sllv':
        regValue[rd] = sllv(regValue[rd], regValue[rt], regValue[rs])
    elif instruction == 'slt':
        regValue[rd] = slt(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'sltu':
        regValue[rd] = sltu(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'sra':
        regValue[rd] = sra(regValue[rd], regValue[rt], int(sa, 2))
    elif instruction == 'srav':
        regValue[rd] = srav(regValue[rd], regValue[rt], regValue[rs])
    elif instruction == 'srl':
        regValue[rd] = srl(regValue[rd], regValue[rt], int(sa, 2))
    elif instruction == 'srlv':
        regValue[rd] = srlv(regValue[rd], regValue[rt], regValue[rs])
    elif instruction == 'sub':
        regValue[rd] = sub(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'subu':
        regValue[rd] = subu(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'xor':
        regValue[rd] = xor(regValue[rd], regValue[rs], regValue[rt])
    elif instruction == 'addi':
        regValue[rt] = addi(regValue[rt], regValue[rs], imm)
    elif instruction == 'addiu':
        regValue[rt] = addiu(regValue[rt], regValue[rs], imm)
    elif instruction == 'andi':
        regValue[rt] = andi(regValue[rt], regValue[rs], imm)
    elif instruction == 'beq':
        pc = beq(regValue[rs], regValue[rt], imm, pc)
    elif instruction == 'bgez':
        pc = bgez(regValue[rs], imm, pc)
    elif instruction == 'bgtz':
        pc = bgtz(regValue[rs], imm, pc)
    elif instruction == 'blez':
        pc = blez(regValue[rs], imm, pc)
    elif instruction == 'bltz':
        pc = bltz(regValue[rs], imm, pc)
    elif instruction == 'bne':
        pc = bne(regValue[rs], regValue[rt], imm, pc)
    elif instruction == 'lb':
        regValue[rt] = lb(regValue[rt], regValue[rs], imm)
    elif instruction == 'lbu':
        regValue[rt] = lbu(regValue[rt], regValue[rs], imm)
    elif instruction == 'lh':
        regValue[rt] = lh(regValue[rt], regValue[rs], imm)
    elif instruction == 'lhu':
        regValue[rt] = lhu(regValue[rt], regValue[rs], imm)
    elif instruction == 'lui':
        regValue[rt] = lui(regValue[rt], imm)
    elif instruction == 'lw':
        regValue[rt] = lw(regValue[rt], regValue[rs], imm)
    elif instruction == 'ori':
        regValue[rt] = ori(regValue[rt], regValue[rs], imm)
    elif instruction == 'sb':
        sb(regValue[rt], regValue[rs], imm)
    elif instruction == 'slti':
        regValue[rt] = slti(regValue[rt], regValue[rs], imm)
    elif instruction == 'sltiu':
        regValue[rt] = sltiu(regValue[rt], regValue[rs], imm)
    elif instruction == 'sh':
        sh(regValue[rt], regValue[rs], imm)
    elif instruction == 'sw':
        sw(regValue[rt], regValue[rs], imm)
    elif instruction == 'xori':
        regValue[rt] = xori(regValue[rt], regValue[rs], imm)
    elif instruction == 'lwl':
        regValue[rt] = lwl(regValue[rt], regValue[rs], imm)
    elif instruction == 'lwr':
        regValue[rt] = lwr(regValue[rt], regValue[rs], imm)
    elif instruction == 'swl':
        swl(regValue[rt], regValue[rs], imm)
    elif instruction == 'swr':
        swr(regValue[rt], regValue[rs], imm)
    elif instruction == 'j':
        pc = j(target)
    elif instruction == 'jal':
        regValue['11111'], pc = jal(target, pc)
    elif instruction == 'syscall':
        syscall(regValue['00010'], regValue['00100'], regValue['00101'], regValue['00110'])

out.close()
inn.close()
