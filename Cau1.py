from sympy import symbols
from sympy.logic.boolalg import And, Or, Not, to_dnf
import re

def split_to_elements(exp_str):
    # Use regex to find all occurrences of variables and their negations
    elements = re.findall(r'~?\w', exp_str)
    return elements

def str_and_to_bool_expr(exp_str):
    str_s = split_to_elements(exp_str)
    for str

def str_to_bool_expr(exp_str):
    str_s = [str_and_to_bool_expr(sub) for sub in exp_str.split('+')]
    print(str_s)

# Định nghĩa các biến
x, y, z, t = symbols('x y z t')

# Chuỗi biểu thức
exp_str="~xz~t+~xyz+~xzt+~y~zt+~xy~t"

# Bến đổi thành hàm bool
f = str_to_bool_expr(exp_str)

# Chuyển đổi hàm Boolean f thành DNF
dnf_form = to_dnf(f, simplify=True)

print(dnf_form)
