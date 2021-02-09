#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#


""" Constants that represent simple type checker error message, i.e. messages
	that do not have any parameters.
"""

NO_RETURN_VALUE_EXPECTED = 'No return value expected'
MISSING_RETURN_STATEMENT = 'Missing return statement'
INVALID_IMPLICIT_RETURN = 'Implicit return in function which does not return'
INCOMPATIBLE_RETURN_VALUE_TYPE = 'Incompatible return value type'
RETURN_VALUE_EXPECTED = 'Return value expected'
NO_RETURN_EXPECTED = 'Return statement in function which does not return'
RECURSIVE_RESULTS_REQUIRED = ("A results type must be provided for recursive functions with one of the following three syntaxes:\n"
                              "def FUNC_NAME(arg1_name:arg1_type, ...) -> RESULT_TYPES\n"
                              "@types('ARG_TYPES', results='RESULT_TYPES')\n"
                              "#$ header function FUNC_NAME(ARG_TYPES) results(RESULT_TYPES)\n")

INCOMPATIBLE_TYPES = 'Incompatible types'
INCOMPATIBLE_TYPES_IN_ASSIGNMENT = 'Incompatible types in assignment'
INCOMPATIBLE_REDEFINITION = 'Incompatible redefinition'
INCOMPATIBLE_REDEFINITION_STACK_ARRAY = 'Cannot change shape of stack array, because it does not support memory reallocation. Avoid redefinition, or use standard heap array.'
STACK_ARRAY_DEFINITION_IN_LOOP = 'Cannot create stack array in loop, because if does not support memory reallocation. Create array before loop, or use standard heap array.'

UNSUPPORTED_ARRAY_RETURN_VALUE = 'Array return arguments are currently not supported'

INCOMPATIBLE_TYPES_IN_STR_INTERPOLATION = 'Incompatible types in string interpolation'
MUST_HAVE_NONE_RETURN_TYPE = 'The return type of "{}" must be None'
INVALID_TUPLE_INDEX_TYPE = 'Invalid tuple index type'
TUPLE_INDEX_OUT_OF_RANGE = 'Tuple index out of range'
ITERABLE_EXPECTED = 'Iterable expected'
INVALID_SLICE_INDEX = 'Slice index must be an integer or None'
CANNOT_INFER_LAMBDA_TYPE = 'Cannot infer type of lambda'
CANNOT_INFER_ITEM_TYPE = 'Cannot infer iterable item type'
CANNOT_ACCESS_INIT = 'Cannot access "__init__" directly'
CANNOT_ASSIGN_TO_METHOD = 'Cannot assign to a method'
CANNOT_ASSIGN_TO_TYPE = 'Cannot assign to a type'
INCONSISTENT_ABSTRACT_OVERLOAD = \
    'Overloaded method has both abstract and non-abstract variants'
READ_ONLY_PROPERTY_OVERRIDES_READ_WRITE = \
    'Read-only property cannot override read-write property'
FORMAT_REQUIRES_MAPPING = 'Format requires a mapping'
RETURN_TYPE_CANNOT_BE_CONTRAVARIANT = "Cannot use a contravariant type variable as return type"
FUNCTION_PARAMETER_CANNOT_BE_COVARIANT = "Cannot use a covariant type variable as a parameter"
INCOMPATIBLE_IMPORT_OF = "Incompatible import of"
FUNCTION_TYPE_EXPECTED = "Function is missing a type annotation"
ONLY_CLASS_APPLICATION = "Type application is only supported for generic classes"
RETURN_TYPE_EXPECTED = "Function is missing a return type annotation"
ARGUMENT_TYPE_EXPECTED = "Function is missing a type annotation for one or more arguments"
KEYWORD_ARGUMENT_REQUIRES_STR_KEY_TYPE = \
    'Keyword argument only valid with "str" key type in call to "dict"'
ALL_MUST_BE_SEQ_STR = 'Type of __all__ must be {}, not {}'
INVALID_TYPEDDICT_ARGS = \
    'Expected keyword arguments, {...}, or dict(...) in TypedDict constructor'
TYPEDDICT_KEY_MUST_BE_STRING_LITERAL = \
    'Expected TypedDict key to be string literal'
MALFORMED_ASSERT = 'Assertion is always true, perhaps remove parentheses?'
NON_BOOLEAN_IN_CONDITIONAL = 'Condition must be a boolean'
DUPLICATE_TYPE_SIGNATURES = 'Function has duplicate type signatures'
GENERIC_INSTANCE_VAR_CLASS_ACCESS = 'Access to generic instance variables via class is ambiguous'
CANNOT_ISINSTANCE_TYPEDDICT = 'Cannot use isinstance() with a TypedDict type'
CANNOT_ISINSTANCE_NEWTYPE = 'Cannot use isinstance() with a NewType type'
BARE_GENERIC = 'Missing type parameters for generic type'
IMPLICIT_GENERIC_ANY_BUILTIN = 'Implicit generic "Any". Use \'{}\' and specify generic parameters'
INCOMPATIBLE_TYPEVAR_VALUE = 'Value of type variable "{}" of {} cannot be {}'
INCOMPATIBLE_TYPEVAR_TO_FUNC = 'TypeError: ufunc "{}" not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule \'safe\''
UNSUPPORTED_ARGUMENT_2_FOR_SUPER = 'Unsupported argument 2 for "super"'
WRONG_NUMBER_OUTPUT_ARGS = 'Number of output arguments does not match number of provided variables'
INDEXED_TUPLE = 'Tuples must be indexed with constant integers for the type inference to work'
LIST_OF_TUPLES = 'Cannot create list of non-homogeneous tuples'

UNUSED_DECORATORS = 'Decorator(s) not used'

UNDEFINED_LAMBDA_VARIABLE = 'Unknown variable(s) in lambda function'
UNDEFINED_LAMBDA_FUNCTION = 'Unknown function in lambda function'

UNDEFINED_INTERFACE_FUNCTION = 'Interface function(s) not found'

UNDEFINED_WITH_ACCESS = 'The __enter__ or __exit__ method for the with construct cannot be found'

VARARGS = 'An undefined number of input arguments is not covered by Pyccel'

# sympy limitation
SYMPY_RESTRICTION_DICT_KEYS = 'sympy does not allow dictionary keys to be strings'

# Pyccel limitation
PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY = 'Uncovered try/except/finally statements by Pyccel'
PYCCEL_RESTRICTION_RAISE = 'Uncovered raise statement by Pyccel'
PYCCEL_RESTRICTION_YIELD = 'Uncovered yield statement by Pyccel'
PYCCEL_RESTRICTION_IS_ISNOT = 'Only booleans and None are allowed as rhs/lhs for is/isnot statement'
PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE = 'Cannot translate "is" comparison, because function "id()" is not implemented for basic types "int", "float", "complex" and "str". Please use "==" for comparison instead.'
PYCCEL_RESTRICTION_IMPORT = 'Import must be inside a def statement or a module'
PYCCEL_RESTRICTION_IMPORT_IN_DEF = 'Only From Import is allowed inside a def statement'
PYCCEL_RESTRICTION_IMPORT_STAR = 'import * not allowed'
PYCCEL_RESTRICTION_OPTIONAL_NONE = 'Variables cannot be equal to None unless they are optional arguments and None is the default value'
PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX = 'Pyccel has encountered syntax that it does not recognise'
PYCCEL_RESTRICTION_TODO = "Pyccel has encountered syntax that has not been implemented yet. Please create an issue at https://github.com/pyccel/pyccel/issues and provide a small example of your problem. Do not forget to specify your target language"
PYCCEL_RESTRICTION_MULTIPLE_COMPARISONS = 'Uncovered multi operator comparison statement'
PYCCEL_RESTRICTION_LIST_COMPREHENSION_ASSIGN = "The result of a list comprehension expression must be saved in a variable"
PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE = 'Could not deduce the size of this list comprehension. If you believe this expression is simple then please create an issue at https://github.com/pyccel/pyccel/issues and provide a small example of your problem.'
PYCCEL_RESTRICTION_LIST_COMPREHENSION_LIMITS = 'Pyccel cannot handle this list comprehension. This is because there are occasions where the upper bound is smaller than the lower bound for variable {}'
PYCCEL_RESTRICTION_INHOMOG_LIST = 'Inhomogeneous lists are not supported by Pyccel. Please use a tuple'

# Fortran limitation
FORTRAN_ALLOCATABLE_IN_EXPRESSION = 'An allocatable function cannot be used in an expression'
FORTRAN_RANDINT_ALLOCATABLE_IN_EXPRESSION = "Numpy's randint function does not have a fortran equivalent. It can be expressed as '(high-low)*rand(size)+low' using numpy's rand, however allocatable function cannot be used in an expression"
FORTRAN_ELEMENTAL_SINGLE_ARGUMENT = 'Elemental functions are defined as scalar operators, with a single dummy argument'

# other Pyccel messages
PYCCEL_INVALID_HEADER = 'Annotated comments must start with omp, acc or header'
PYCCEL_MISSING_HEADER = 'Cannot find associated header'
MACRO_MISSING_HEADER_OR_FUNC = 'Cannot find associated header/FunctionDef to macro'
PYCCEL_UNFOUND_IMPORTED_MODULE = 'Unable to import'
FOUND_DUPLICATED_IMPORT = 'Duplicated import '
PYCCEL_UNEXPECTED_IMPORT = 'Pyccel has not correctly handled "import module" statement. Try again with "from module import function" syntax'

IMPORTING_EXISTING_IDENTIFIED = \
        'Trying to import an identifier that already exists in the namespace. Hint: use import as'

UNDEFINED_FUNCTION = 'Undefined function'
UNDEFINED_VARIABLE = 'Undefined variable'
UNDEFINED_INDEXED_VARIABLE = 'Undefined indexed variable'
UNDEFINED_IMPORT_OBJECT = 'Could not find {} in imported module {}'

REDEFINING_VARIABLE = 'Variable already defined'

INVALID_FOR_ITERABLE = 'Invalid iterable object in For statement'

INVALID_FILE_DIRECTORY = 'No file or directory of this name'
INVALID_FILE_EXTENSION = 'Wrong file extension. Expecting `py` of `pyh`, but found'
INVALID_PYTHON_SYNTAX = 'Python syntax error'

# ARRAY ERRORS
ASSIGN_ARRAYS_ONE_ANOTHER = 'Arrays which own their data cannot become views on other arrays'
ARRAY_ALREADY_IN_USE = 'Attempt to reallocate an array which is being used by another variable'
INVALID_POINTER_REASSIGN = 'Attempt to give data ownership to a pointer'
INVALID_INDICES = 'only integers and slices (`:`) are valid indices'

# warnings
UNDEFINED_INIT_METHOD = 'Undefined `__init__` method'
FOUND_SYMBOLIC_ASSIGN = 'Found symbolic assignment [Ignored]'
FOUND_IS_IN_ASSIGN = 'Found `is` statement in assignment [Ignored]'
ARRAY_REALLOCATION = 'Array redefinition may cause memory reallocation at runtime'
ARRAY_DEFINITION_IN_LOOP = 'Array definition in for loop may cause memory reallocation at each cycle. Consider creating the array before the loop'
TEMPLATE_IN_UNIONTYPE = 'Cannot use templates in a union type'
DUPLICATED_SIGNATURE = 'Same signature defined for the same function multiple times'
