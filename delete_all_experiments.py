"""
Deletes all experiments from the Crayon server.
"""

import pycrayon

def main():
    """ Asks users whether they want to delete all experiments and then deletes."""
    crayon_client = pycrayon.CrayonClient("localhost")

    choice = raw_input("Are you sure? ")
    if choice == 'y':
        crayon_client.remove_all_experiments()
    else:
        print("Not deleting")

if __name__ == "__main__":
    main()
