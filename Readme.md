# Git Commands Summary

To clone the repository:

```bash
git clone https://github.com/TongHan96/AMID.git
cd AMID
```

To add and commit the `input` directory to `src/GAT`:

```bash
git add src/GAT/input
git commit -m "Added input directory to src/GAT"
```

To push changes to the `main` branch:

```bash
git push origin main
```


If there are conflicts...

1. **Fetch Remote Changes**  
   Before you merge, fetch the changes from the remote repository. This allows you to see the changes without merging them immediately.
   ```bash
   git fetch origin
   ```

2. **Check for Changes**  
   It's a good practice to see the commits that are different between your branch and the remote.
   ```bash
   git log main..origin/main
   ```
   This will show the commits that are in the remote repository but not in your local main branch.

3. **Merge or Rebase**  
   Depending on your workflow and how you like to keep your commit history, you can either merge or rebase:

   - **Merge**: This will create a new commit in your local main branch that merges in the changes from the remote main branch.
     ```bash
     git merge origin/main
     ```

   - **Rebase**: This will add your local commits on top of the remote's commits. This keeps a linear commit history, but be careful as it rewrites commit history.
     ```bash
     git rebase origin/main
     ```

   If there are merge conflicts during either process, you will need to resolve them before proceeding.

4. **Push Changes**  
   Once you've integrated the remote changes, you can push your changes to GitHub.
   ```bash
   git push origin main
   ```

In the future, before making changes and pushing them, it's a good habit to always pull the latest changes from the remote repository to ensure you're working with the most recent codebase. This reduces the chances of encountering such conflicts.
