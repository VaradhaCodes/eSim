import re
import sys

def remove_symbol(content, modelname):
    # We want to remove the block starting with (symbol "modelname" 
    # up to the matching closing parenthesis.
    # A simple regex won't work well due to nested parenthesis.
    # Let's write a simple parenthesis matcher.
    
    # Find start of symbol
    start_idx = content.find(f'(symbol "{modelname}"')
    if start_idx == -1:
        return content
        
    # Find matching closing parenthesis
    paren_count = 0
    in_string = False
    escape = False
    
    for i in range(start_idx, len(content)):
        c = content[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if c == '(':
                paren_count += 1
            elif c == ')':
                paren_count -= 1
                if paren_count == 0:
                    # Found the end
                    end_idx = i
                    # return content without this block
                    # also strip trailing newlines of the removed block if any
                    res = content[:start_idx] + content[end_idx+1:]
                    return res
                    
    return content

# Test it
test_content = """(kicad_symbol_lib (version 20211014) (generator kicad_symbol_editor)
  
(symbol "cpu" (pin_names (offset 1.016)) (in_bom yes) (on_board yes)
(property "Reference" "U" (id 0) (at 12 15 0)(effects (font (size 1.524 1.524))))
(symbol "cpu_0_1" (rectangle (start 0 0) (end 1 1)))
(symbol "cpu_1_1" (pin input line (at 0 0 0) (length 1)))
)
(symbol "other" (pin_names)
(property)
)
)"""

print("Original length:", len(test_content))
res = remove_symbol(test_content, "cpu")
print("After removing cpu length:", len(res))
if "(symbol \"cpu\"" not in res and "(symbol \"other\"" in res:
    print("Test passed")
else:
    print("Test failed")
