# Kubernetees- INSTRUCTIONS
---

## 🚀 Step 1: Clone the Repository
### 1️⃣ Open WSL Terminal.
### 2️⃣ Navigate to your home directory:
cd ~
### 3️⃣ Clone the GitHub repository:
git clone git@github.com:Haris0010/Kubernetees-.git
### 4️⃣ Move into the project folder:
cd Kubernetees-
### ✅ Now you have the project files in WSL.
---

## 🚀 Step 2: Set Up Git Identity (If You Haven’t Already)
Git requires your name and email to track changes. 
### Run these commands:
#### git config --global user.name "Your Name"
#### git config --global user.email "your_email@example.com"  (GITHUB EMAIL)
### Verify the settings:
git config --global --list
## ✅ Now Git will recognize your commits. Now can do push pull
---

## 🚀 Step 3: Set Up SSH Authentication (If You Haven’t Already)
### 1️⃣ Check if you already have an SSH key:
ls ~/.ssh/id_rsa.pub
### If the file exists, skip to Step 4.
### If not, generate a new SSH key:
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"   (GITHUB EMAIL)
### Press ENTER for all prompts.
### 2️⃣ Copy your SSH key:
cat ~/.ssh/id_rsa.pub 
### 3️⃣ Add the key to GitHub:
Go to GitHub → Settings → SSH and GPG keys.
#### Click "New SSH Key", paste the copied key, and save.
### 4️⃣ Test the SSH connection:
ssh -T git@github.com
### ✅ If successful, you’ll see:
Hi <your-username>! You've successfully authenticated, but GitHub does not provide shell access.
---

## 🚀 Step 4: Pull the Latest Updates
Before starting work, always pull the latest version of the project:
### git pull origin main
### ✅ Now your local project is up to date with GitHub.
---

## 🚀 Step 5: Add the Dataset to WSL
Since the dataset is ignored by GitHub, each team member must manually copy it to WSL.
### 1️⃣ Move the dataset from Windows to WSL:
mv /mnt/c/Users/<YourName>/Downloads/<file_name>.csv ~/Kubernetees-/hdb-price-prediction/data/
### 🔹 Replace <YourName> with your actual Windows username.
## 2️⃣ Verify the dataset is in the right location:
ls ~/Kubernetees-/hdb-price-prediction/data/
## ✅ Now your dataset is inside WSL and ready for processing.
---

## 🚀 Step 6: Edit a File in the Project
To edit a file in WSL, follow these steps:
### 1️⃣ Navigate to the Project Folder
If you are not already inside the project directory, move into it:
#### cd ~/Kubernetees-/hdb-price-prediction
### 2️⃣ Open the File for Editing
You can edit files using different methods:
### Use Nano (Simple Terminal Editor)
### nano <path-to-file>
### 🔹 Example (for preprocess.py):
### nano preprocessing/preprocess.py
### To save changes: Press CTRL + X, then Y, then ENTER.
---

## 🚀 Step 7: Save & Push Changes to GitHub
Once you have edited and saved a file, you need to push your changes.
### 1️⃣ Check What Has Changed
git status
#### 🔹 This will show all modified files.
### 2️⃣ Stage the Modified File
git add <path-to-file>
#### 🔹 Example:
git add preprocessing/preprocess.py
##### Or stage all modified files:
git add .
### 3️⃣ Commit the Changes
git commit -m "Describe what you changed"
#### 🔹 Example:
git commit -m "Improved data cleaning in preprocess.py"
### 4️⃣ Push the Changes to GitHub
git push origin main
---

## 🚀 Step 8: Pull Updates from Other Team Members
Before making new changes, always pull the latest version to avoid conflicts:
### git pull origin main
### ✅ Now your local files match GitHub.
