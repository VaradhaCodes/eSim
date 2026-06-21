def append_symbol(content, new_symbol):
    content = content.rstrip()
    if content.endswith(')'):
        content = content[:-1].rstrip()
    else:
        # If it doesn't end with ), maybe it's completely empty?
        if not content:
            content = "(kicad_symbol_lib (version 20211014) (generator kicad_symbol_editor)\n"
    
    # Append the new symbol, and re-add the closing )
    res = content + "\n\n" + new_symbol + "\n)\n"
    return res

test_content = "(kicad_symbol_lib (version 20211014) (generator kicad_symbol_editor)\n)"
print(append_symbol(test_content, '(symbol "cpu" (pin_names) )'))
