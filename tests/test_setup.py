import os

def test_package_structure():
    assert os.path.exists("src/Heart")
    assert os.path.exists("setup.py")
    assert os.path.exists("requirements.txt")
    print("Package structure verification passed.")

if __name__ == '__main__':
    test_package_structure()
