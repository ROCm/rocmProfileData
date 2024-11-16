import os
import pathlib
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
recipes = list(pathlib.Path(test_dir, '..', 'cookbook').resolve().glob('*.ipy'))

@pytest.mark.parametrize('recipe', recipes, ids=[os.path.basename(r) for r in recipes])
def test_recipe(recipe):
    cmd = 'ipython ' + str(recipe)
    assert not os.system(cmd)

