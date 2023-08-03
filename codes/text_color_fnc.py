# Create a class and add a bunch of functions to it to color the text in the terminal.
# Use the following colors: Red, Green, Yellow, Blue, Magenta, Cyan, White, Black, Aqua, Orange,
# Purple, Pink, Grey, and Brown.


class text_color:
    def red_text(self, text):
        return "\033[91m{}\033[00m".format(text)

    def green_text(self, text):
        return "\033[92m{}\033[00m".format(text)

    def yellow_text(self, text):
        return "\033[93m{}\033[00m".format(text)

    def blue_text(self, text):
        return "\033[94m{}\033[00m".format(text)

    def magenta_text(self, text):
        return "\033[95m{}\033[00m".format(text)

    def cyan_text(self, text):
        return "\033[96m{}\033[00m".format(text)

    def white_text(self, text):
        return "\033[97m{}\033[00m".format(text)

    def black_text(self, text):
        return "\033[98m{}\033[00m".format(text)

    def aqua_text(self, text):
        return "\033[99m{}\033[00m".format(text)

    def orange_text(self, text):
        return "\033[910m{}\033[00m".format(text)

    def purple_text(self, text):
        return "\033[911m{}\033[00m".format(text)

    def pink_text(self, text):
        return "\033[912m{}\033[00m".format(text)

    def grey_text(self, text):
        return "\033[913m{}\033[00m".format(text)

    def brown_text(self, text):
        return "\033[914m{}\033[00m".format(text)

    def turquoise_text(self, text):
        return "\033[915m{}\033[00m".format(text)

    def lime_text(self, text):
        return "\033[916m{}\033[00m".format(text)

    def bold_text(self, text):
        return "\033[1m{}\033[00m".format(text)

    def underline_text(self, text):
        return "\033[4m{}\033[00m".format(text)

    def blink_text(self, text):
        return "\033[5m{}\033[00m".format(text)

    def reverse_text(self, text):
        return "\033[7m{}\033[00m".format(text)

    def invisible_text(self, text):
        return "\033[8m{}\033[00m".format(text)

    def strikethrough_text(self, text):
        return "\033[9m{}\033[00m".format(text)

    def double_underline_text(self, text):
        return "\033[21m{}\033[00m".format(text)

    def double_intensity_text(self, text):
        return "\033[22m{}\033[00m".format(text)


# Create a class and add a bunch of functions to it to color the background in the terminal.
# Use the following colors: Red, Green, Yellow, Blue, Magenta, Cyan, White, Black, Aqua, Orange,
# Purple, Pink, Grey, and Brown, Turquoise, and Lime.


class background_color:
    def red_background(self, text):
        return "\033[41m{}\033[00m".format(text)

    def green_background(self, text):
        return "\033[42m{}\033[00m".format(text)

    def yellow_background(self, text):
        return "\033[43m{}\033[00m".format(text)

    def blue_background(self, text):
        return "\033[44m{}\033[00m".format(text)

    def magenta_background(self, text):
        return "\033[45m{}\033[00m".format(text)

    def cyan_background(self, text):
        return "\033[46m{}\033[00m".format(text)

    def white_background(self, text):
        return "\033[47m{}\033[00m".format(text)

    def black_background(self, text):
        return "\033[48m{}\033[00m".format(text)

    def aqua_background(self, text):
        return "\033[49m{}\033[00m".format(text)

    def orange_background(self, text):
        return "\033[410m{}\033[00m".format(text)

    def purple_background(self, text):
        return "\033[411m{}\033[00m".format(text)

    def pink_background(self, text):
        return "\033[412m{}\033[00m".format(text)

    def grey_background(self, text):
        return "\033[413m{}\033[00m".format(text)

    def brown_background(self, text):
        return "\033[414m{}\033[00m".format(text)

    def turquoise_background(self, text):
        return "\033[415m{}\033[00m".format(text)

    def lime_background(self, text):
        return "\033[416m{}\033[00m".format(text)

    def bold_background(self, text):
        return "\033[1m{}\033[00m".format(text)

    def underline_background(self, text):
        return "\033[4m{}\033[00m".format(text)

    def blink_background(self, text):
        return "\033[5m{}\033[00m".format(text)

    def reverse_background(self, text):
        return "\033[7m{}\033[00m".format(text)

    def invisible_background(self, text):
        return "\033[8m{}\033[00m".format(text)


"""
# Create a sample text to test the functions.
sample_text = "This is a sample text."

# Run the functions.
text_color = text_color()
background_color = background_color()

print(text_color.red_text(sample_text))
print(text_color.green_text(sample_text))
print(text_color.yellow_text(sample_text))
print(text_color.blue_text(sample_text))
print(text_color.magenta_text(sample_text))
print(text_color.cyan_text(sample_text))
print(text_color.white_text(sample_text))
print(text_color.black_text(sample_text))
print(text_color.aqua_text(sample_text))
print(text_color.orange_text(sample_text))
print(text_color.purple_text(sample_text))
print(text_color.pink_text(sample_text))
print(text_color.grey_text(sample_text))
print(text_color.brown_text(sample_text))
print(text_color.turquoise_text(sample_text))
print(text_color.lime_text(sample_text))
print(text_color.bold_text(sample_text))
print(text_color.underline_text(sample_text))
print(text_color.blink_text(sample_text))
print(text_color.reverse_text(sample_text))
print(text_color.invisible_text(sample_text))
print(text_color.strikethrough_text(sample_text))
print(text_color.double_underline_text(sample_text))
print(text_color.double_intensity_text(sample_text))

# Write a case where you use all the functions in a different file.
# Create a new file and import the functions from the previous file.
# Run the functions.
"""
