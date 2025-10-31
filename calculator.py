def calculator():
    print("this is a basic calculator\n")
    print("choose your best option:")
    print("1:Addition (+)")
    print("2:Substraction (-)")
    print("3:Multiplication (*)")
    print("4:Division (:)")
    
    choice=input("your choice:(1/2/3/4): ")
    
    if choice in['1','2','3','4']:
        try:
            num1=float(input("enter first number: "))
            num2=float(input("enter second number: "))
            if choice=='1':
               print(f"{num1} + {num2} = {num1 + num2}")
            elif choice=='2':
             print(f"{num1} - {num2} = {num1 - num2}")
            elif choice=='3':
             print(f"{num1} * {num2} = {num1 * num2}")
            elif choice=='4':
              if num2!=0:
                print(f"{num1} / {num2} = {num1 / num2}")
              else:
                print("division by zero is impossible")
        except ValueError:
            print("Invalid input! Please enter numbers only.")        
    else:
              print("invalid choice")
calculator()