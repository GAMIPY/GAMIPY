import unittest

from GAMIPY.population.gene import Gene


class TestGene(unittest.TestCase):
    def setUp(self):
        self.gene = Gene(orig_location=(1, 2), m_location=(1, 2), name="test_gene", data=42, dtype="num")

    def test_initialization(self):
        self.assertEqual(self.gene.location, (1, 2))
        self.assertEqual(self.gene._m_location, (1, 2))
        self.assertEqual(self.gene.name, "test_gene")
        self.assertEqual(self.gene.data, 42)
        self.assertEqual(self.gene.dtype, "num")

    def test_wrong_positional_argument_order_now_raises_type_error(self):
        # This is exactly the call shape Arithmetic/Blend
        # crossover used to use (now fixed) - Gene(location, name, data,
        # dtype), whose positional arguments didn't line up with Gene's
        # signature. Arguments are now keyword-only, so any positional call
        # fails loudly at construction time instead of silently producing a
        # corrupted gene.
        location = (3, "col_a")
        name = "col_a"  # lands in m_location's slot - not a tuple
        data = 12.5
        dtype = "cat"

        with self.assertRaises(TypeError):
            Gene(location, name, data, dtype)

    def test_non_tuple_orig_location_raises_type_error(self):
        with self.assertRaises(TypeError):
            Gene(orig_location="not_a_tuple", m_location=(0, 0), name="x", data=1, dtype="num")

    def test_non_tuple_m_location_raises_type_error(self):
        with self.assertRaises(TypeError):
            Gene(orig_location=(0, 0), m_location="not_a_tuple", name="x", data=1, dtype="num")

    def test_invalid_dtype_raises_value_error(self):
        # Gene.__init__ now validates dtype against
        # {"num", "cat"} instead of silently accepting anything.
        with self.assertRaises(ValueError):
            Gene(orig_location=(0, 0), m_location=(0, 0), name="x", data=1, dtype="not_a_real_dtype")

    def test_every_argument_is_required(self):
        complete = dict(orig_location=(0, 0), m_location=(0, 0), name="x", data=1, dtype="num")
        for missing in complete:
            with self.subTest(missing=missing):
                with self.assertRaises(TypeError):
                    Gene(**{k: v for k, v in complete.items() if k != missing})

    def test_valid_cat_dtype_is_accepted(self):
        gene = Gene(orig_location=(0, 0), m_location=(0, 0), name="x", data="red", dtype="cat")
        self.assertEqual(gene.dtype, "cat")
