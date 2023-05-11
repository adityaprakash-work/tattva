#------------------------------------------------------------------------------

#---CONSTANTS------------------------------------------------------------------


#---ERRORS---------------------------------------------------------------------
class ExpectationError(Exception):
    pass

#---GUARD BLOCKS---------------------------------------------------------------
class ExpectationGuard:
    """
    A context manager that provides a guard clause for checking expectations.

    The ExpectationGuard is a context manager that allows you to set up a guard 
    clause for checking expectations. It takes an error object and a severity 
    level as input, and raises the error if the expectation is not met, up to the 
    specified severity level.

    Usage:
    ```
    with ExpectationGuard(error, severity) as c:
        # Perform checks and set expectation
        c(expectation)
    ```

    Args:
        error (Exception): The error object to raise if the expectation is not 
            met.
        severity (int): The severity level of the guard clause. The guard clause 
            will raise the error if the expectation is not met, up to this severity 
            level.

    Attributes:
        error (Exception): The error object to raise if the expectation is not 
            met.
        severity (int): The severity level of the guard clause.
        checks (int): The number of times the `check` method has been called.

    Methods:
        check(expectation: bool)
            Check the expectation and raise the error if it is not met, up to the 
            specified severity level.
    """
    def __init__(self, error: Exception, severity: int):
        self.error = error
        self.severity = severity
        self.checks = 0

    def __enter__(self):
        
        return self.check
    
    def check(self, expectation: bool):
        if expectation == 0 and self.checks < self.severity:
            raise self.error('Guard clause fail')
        else:
            self.checks += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        pass




