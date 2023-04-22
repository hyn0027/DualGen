def main():
    with open("./identical_dict.txt", "w") as f:
        for i in range(1, 400):
            f.write(str(i) + " 1\n")


if __name__ == '__main__':
   main()
