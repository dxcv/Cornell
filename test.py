
def postfix_to_infix( expression):
    expression = expression.strip()
    operators =['+','-','*','/','^']
    level1 = ['+','-']
    level2 = ['*','/']
    level3 = ['^']
    endd = len(expression)
    if (ord(expression[0])<ord('a'))|(ord(expression[1])<ord('a')):
        return 'invalid'
    if ord(expression[2])>=ord('a'):
        return 'invalid'
    if ord(expression[endd-1])>=ord('a'):
        return 'invalid'
    ops =[]
    for i in range(2,endd,2):
        if expression[i] not in operators:
            return 'invalid'
            break
        else:
            ops.append(expression[i])
    oprands =[]
    for i in range(1,endd,2):
        if (ord(expression[i])>122)|(ord(expression[i])<97):
            return 'invalid'
            break
        else:
            oprands.append()

    for i in range(0,len(oprands)):
        oprands.insert(i+1,ops[i])
    return operands