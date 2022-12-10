import numpy as np


def first_digit_prob(digit):
    return np.log10(1 + (1 / digit))


def second_digit_prob(digit):
    return sum([np.log10(1 + (1 / int(str(first_digit) + str(digit)))) for first_digit in range(1, 10)])


def generate_benfords_nbr(nbr_digits, nbr_values_to_generate=100):
    first_digit_probs = {d: first_digit_prob(digit=d) for d in range(1, 10)}
    second_digit_probs = {d: second_digit_prob(digit=d) for d in range(10)}
    if nbr_digits == 1:
        first_digit = [
            np.random.choice(
                a=list(first_digit_probs),
                p=list(first_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        return first_digit
    elif nbr_digits == 2:
        first_digit = [
            np.random.choice(
                a=list(first_digit_probs),
                p=list(first_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        second_digit = [
            np.random.choice(
                a=list(second_digit_probs),
                p=list(second_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        return [int(str(fd) + str(second_digit[idx])) for idx, fd in enumerate(first_digit)]
    elif nbr_digits == 3:
        first_digit = [
            np.random.choice(
                a=list(first_digit_probs),
                p=list(first_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        second_digit = [
            np.random.choice(
                a=list(second_digit_probs),
                p=list(second_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        third_digit = [
            np.random.choice(
                a=[d for d in range(10)],
                p=[0.1 for _ in range(10)]
            )
            for _ in range(nbr_values_to_generate)
        ]
        return [int(str(fd) + str(second_digit[idx]) + str(third_digit[idx])) for idx, fd in enumerate(first_digit)]
    elif nbr_digits == 4:
        first_digit = [
            np.random.choice(
                a=list(first_digit_probs),
                p=list(first_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        second_digit = [
            np.random.choice(
                a=list(second_digit_probs),
                p=list(second_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        last_two_digits = [
            np.random.choice(
                a=[d for d in range(100)],
                p=[0.01 for _ in range(100)]
            )
            for _ in range(nbr_values_to_generate)
        ]
        return [int(str(fd) + str(second_digit[idx]) + str(last_two_digits[idx])) for idx, fd in enumerate(first_digit)]
    elif nbr_digits > 4:
        first_digit = [
            np.random.choice(
                a=list(first_digit_probs),
                p=list(first_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        second_digit = [
            np.random.choice(
                a=list(second_digit_probs),
                p=list(second_digit_probs.values())
            )
            for _ in range(nbr_values_to_generate)
        ]
        last_two_digits = [
            np.random.choice(
                a=[d for d in range(100)],
                p=[0.01 for _ in range(100)]
            )
            for _ in range(nbr_values_to_generate)
        ]
        middle_digits = [
            ''.join([
                str(np.random.choice(
                    a=[d for d in range(10)],
                    p=[0.1 for _ in range(10)]
                ))
                for _ in range(nbr_digits - 4)
            ])
            for _ in range(nbr_values_to_generate)
        ]
        return [
            int(str(fd)
                + str(second_digit[idx])
                + middle_digits[idx]
                + str(last_two_digits[idx]))
            for idx, fd in enumerate(first_digit)
        ]
    else:
        raise Exception("Invalid nbr_digits")


numbers = generate_benfords_nbr(nbr_digits=1, nbr_values_to_generate=10)
print(numbers)
numbers = generate_benfords_nbr(nbr_digits=2, nbr_values_to_generate=10)
print(numbers)
numbers = generate_benfords_nbr(nbr_digits=3, nbr_values_to_generate=10)
print(numbers)
numbers = generate_benfords_nbr(nbr_digits=4, nbr_values_to_generate=10)
print(numbers)
numbers = generate_benfords_nbr(nbr_digits=6, nbr_values_to_generate=10)
print(numbers)


