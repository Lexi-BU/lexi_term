from text_color_fnc import text_color as tc

import importlib


# Import all the functions from text_color_fnc.py
# and assign it to a variable named tc.
#
# This is the same as:
# from text_color_fnc import text_color
# tc = text_color()
#
# Now, we can use the functions from text_color_fnc.py
# by using tc.<function_name>

text = "Hello World!"

print(tc().red_text(text))
