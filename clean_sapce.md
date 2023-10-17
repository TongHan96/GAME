## Cleaning up Space in Amazon SageMaker

### 1. Check Disk Usage:
```bash
df -h
```

### 2. Navigate to SageMaker directory:
```bash
cd /home/ec2-user/SageMaker
```

### 3. List the contents:
```bash
ls
```

### 4. Find Large Files and Directories:
```bash
du -h --max-depth=1 .
```

### 5. Clean the `.Trash-1000` Directory:
If the `.Trash-1000` directory is consuming a lot of space, it's the "Trash" or "Recycle bin" of your system. Deleting its contents will free up space:
```bash
rm -rf .Trash-1000
```

### 6. Recheck Disk Usage:
```bash
du -h --max-depth=1 .
```

### 7. Return to Home Directory:
```bash
cd ~
```

### 8. Verify Overall Disk Usage:
```bash
df -h
```

---

Remember to be cautious when deleting files or directories, and always ensure you have backups of any important data before performing any cleanup actions.
