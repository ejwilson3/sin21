# Cosine rule derivation:
# a^2 = h^2 + (c-x)^2
# b^2 = h^2 + x^2
# cos(A) = x/b
# cos(B) = (c-x)/a
# a^2 = b^2 + c^2 - 2bc*cos(A)
# a^2 = b^2 + c^2 - 2bc*(x)/b
# a^2 = b^2 + c^2 - 2cx
# b^2 = a^2 + c^2 - 2ac*cos(B)
# b^2 = a^2 + c^2 - 2ac*(c-x)/a
# b^2 = a^2 + c^2 - 2c*(c-x)
# a^2 = h^2 + x^2 + c^2 - 2cx

# ACTUAL derivation:
# cos(A) = x/b
# x = b*cos(A)
# c-x = c-b*cos(A)
# sin(A) = h/b
# h = b*sin(A)
# from pythagoras
# a^2 = h^2 + (c-x)^2
# a^2 = (b*sin(A))^2 + (c - b*cos(A))^2
# a^2 = b^2*(sin(A)^2 + cos(A)^2) + c^2 - cb*cos(A)
# a^2 = b^2 + c^2 - cb*cos(A)

# find x using cosine rule:
# x = b*cos(A)
# a^2 = b^2 + c^2 - 2bc*cos(A)
# cos(A) = (b^2 + c^2 - a^2)/(2bc)
# x = (b^2 + c^2 - a^2)/(2c)
