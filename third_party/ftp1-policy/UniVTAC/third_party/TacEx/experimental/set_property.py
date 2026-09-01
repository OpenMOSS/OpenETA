from pxr import Gf, Sdf, Usd, UsdGeom

"""
Find all relevant data types at: https://openusd.org/release/api/_usd__page__datatypes.html
"""


def create_float_attribute(prim: Usd.Prim, attribute_name: str) -> Usd.Attribute:
    """Creates attribute for a prim that holds a float.
    See: https://openusd.org/release/api/class_usd_prim.html
    Args:
        prim (Usd.Prim): A Prim for holding the attribute.
        attribute_name (str): The name of the attribute to create.
    Returns:
        Usd.Attribute: An attribute created at specific prim.
    """
    attr: Usd.Attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float)
    return attr


def create_vector_attribute(prim: Usd.Prim, attribute_name: str) -> Usd.Attribute:
    """Creates attribute for a prim that holds a vector.
    See: https://openusd.org/release/api/class_usd_prim.html
    Args:
        prim (Usd.Prim): A Prim for holding the attribute.
        attribute_name (str): The name of the attribute to create.
    Returns:
        Usd.Attribute: An attribute created at specific prim.
    """
    attr: Usd.Attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float3)
    return attr


#############
# Full Usage
#############

# Create an in-memory Stage
stage: Usd.Stage = Usd.Stage.CreateInMemory()

# Create a prim named /World (type Xform) and make it the default prim.
prim_path = "/World"
xform: UsdGeom.Xform = UsdGeom.Xform.Define(stage, prim_path)
prim: Usd.Prim = xform.GetPrim()
stage.SetDefaultPrim(prim)

# Create a float attribute on /World
float_attr: Usd.Attribute = create_float_attribute(prim, "my_float_attr")

# Create a vector attribute on /World
vector_attr: Usd.Attribute = create_vector_attribute(prim, "my_vector_attr")

# Set and query values
print(float_attr.Get())
float_attr.Set(0.1)
print(float_attr.Get())

vector_value: Gf.Vec3f = Gf.Vec3f(0.1, 0.2, 0.3)
print(vector_attr.Get())
vector_attr.Set(vector_value)
print(vector_attr.Get())

# Optionally preview the usd
# print(stage.GetRootLayer().ExportToString())
