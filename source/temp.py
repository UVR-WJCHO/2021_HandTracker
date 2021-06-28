
lsb = 200
msb = 10

lsb = (lsb >> 8) & 0xff
msb = msb & 0xff

print("lsb : ", lsb)
print("msb : ", msb)