"""Fix broken code fences in docs/setup.md."""
path = '/root/forge/docs/setup.md'
content = open(path, 'r').read()

# Fix opening broken fences: \\\x08ash -> ```bash
content = content.replace('\\\x08ash', '```bash')

# Fix closing broken fences: standalone \\ lines -> ```
lines = content.split('\n')
for i, line in enumerate(lines):
    if line == '\\':
        lines[i] = '```'

content = '\n'.join(lines)

# Fix inline escaped package names
content = content.replace('\\onnxruntime\\', '`onnxruntime`')
content = content.replace('\\onnx\\', '`onnx`')

open(path, 'w').write(content)
print('Done')
