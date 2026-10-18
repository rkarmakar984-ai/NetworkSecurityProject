import sys
from networksecurity.logging import logger

# CUSTOM EXCEPTION CLASS SETUP --------------------------------------

class NetworkSecurityException(Exception):
    """Custom exception that records where in the source the error happened."""

    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        if exc_tb is None:
            # Raised outside an 'except' block -- there is no traceback to read.
            self.lineno = None
            self.filename = None
        else:
            self.lineno = exc_tb.tb_lineno
            self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occurred in python script name[{0}] line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, str(self.error_message)
        )