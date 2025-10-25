import os
import shutil
import argparse
import subprocess
import sys

# Define the path to the dataset, relative to this script
DATASET_PATH = "dataset"
ENCODINGS_PATH = "known_faces.pkl"

def list_users():
    """Lists all registered users by their folder names."""
    print("[INFO] Listing all registered users:")
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset folder '{DATASET_PATH}' not found.")
        return

    # Get all items in the dataset path that are directories
    users = [name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))]

    if not users:
        print("  No users are currently registered.")
        return

    for user in users:
        print(f"  - {user}")

def delete_user(name):
    """Deletes a user's data folder and updates the encodings."""
    user_path = os.path.join(DATASET_PATH, name)

    if not os.path.exists(user_path):
        print(f"[ERROR] No user found with the name '{name}'.")
        return

    print(f"[WARNING] You are about to permanently delete all data for '{name}'.")
    # Use input() to ask for confirmation.
    confirm = input("  Are you sure? (y/n): ").lower().strip()

    if confirm == 'y' or confirm == 'yes':
        try:
            # Delete the user's entire folder and all images within
            shutil.rmtree(user_path)
            print(f"[SUCCESS] Successfully deleted user '{name}'.")

            # After deleting, we MUST update the encodings file.
            print("[INFO] Re-running face encoding to update the model...")

            # Find the python executable from the current virtual environment
            # This ensures we use the same environment as main.py
            python_executable = sys.executable

            # Run encode_faces.py as a subprocess
            subprocess.run([python_executable, "encode_faces.py"], check=True)
            print(f"[SUCCESS] Encodings file '{ENCODINGS_PATH}' has been updated.")

        except OSError as e:
            print(f"[ERROR] Could not delete user: {e}")
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to update encodings. Please run 'python encode_faces.py' manually.")
    else:
        print("[INFO] Deletion cancelled.")

def main():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Manage users for the Home Security System.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Create the 'list' command
    list_parser = subparsers.add_parser('list', help='List all registered users.')

    # Create the 'delete' command
    delete_parser = subparsers.add_parser('delete', help='Delete a specific user.')
    delete_parser.add_argument('--name', type=str, required=True, help='The name of the user to delete (must match folder name).')

    args = parser.parse_args()

    # Execute the correct function based on the command
    if args.command == 'list':
        list_users()
    elif args.command == 'delete':
        delete_user(args.name)

if __name__ == "__main__":
    main()
