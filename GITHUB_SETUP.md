# Hướng dẫn Upload Code lên GitHub

## Bước 1: Tạo Repository trên GitHub

1. Đăng nhập vào GitHub: https://github.com
2. Click "New repository" (hoặc vào: https://github.com/new)
3. Đặt tên repository (ví dụ: `ct-slice-interpolation-3dgs`)
4. Chọn Public hoặc Private
5. **KHÔNG** tích "Initialize with README" (vì đã có code sẵn)
6. Click "Create repository"

## Bước 2: Thêm Remote và Push Code

Sau khi tạo repository, GitHub sẽ hiển thị URL. Copy URL đó và chạy các lệnh sau:

### Nếu dùng HTTPS:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Nếu dùng SSH:
```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

## Bước 3: Commit và Push

Chạy script helper:
```bash
# Windows PowerShell
.\setup_github.ps1

# Hoặc chạy thủ công:
git add .
git commit -m "Initial commit: CT slice interpolation via 3D Gaussian Splatting"
git branch -M main
git push -u origin main
```

## Lưu ý

- File `.gitignore` đã được tạo để loại trừ:
  - Dataset lớn (PKG-CT-ORG/)
  - Checkpoints và outputs
  - Python cache files
  - Jupyter checkpoints

- Nếu cần push dataset, bạn có thể dùng Git LFS:
  ```bash
  git lfs install
  git lfs track "*.nii.gz"
  git add .gitattributes
  ```

## Troubleshooting

### Nếu gặp lỗi authentication:
- Dùng Personal Access Token thay vì password
- Hoặc setup SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Nếu muốn thay đổi remote URL:
```bash
git remote set-url origin NEW_URL
```
