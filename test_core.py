import unittest
from types import SimpleNamespace
import numpy as np
from core import GazeModel, heat_grid, validation_error, extract_features

class CoreTests(unittest.TestCase):
    def test_calibration_predicts_unseen_points(self):
        rng=np.random.default_rng(18)
        x=rng.normal(size=(100,8))
        y=np.column_stack([.5+.12*x[:,0]-.08*x[:,1],.5+.13*x[:,2]])
        model=GazeModel().fit(x,y)
        test=rng.normal(size=8)
        expected=[.5+.12*test[0]-.08*test[1],.5+.13*test[2]]
        np.testing.assert_allclose(model.predict(test),expected,atol=.005)
    def test_heatmap_does_not_clip_invalid_points_to_edges(self):
        grid=heat_grid([(0,0),(99,79),(-1,30),(100,40),(20,80)],100,80)
        self.assertEqual(grid.sum(),2)
        self.assertEqual(grid[0,0],1)
        self.assertEqual(grid[-1,-1],1)
    def test_validation_uses_screen_pixels(self):
        self.assertAlmostEqual(validation_error([.6,.5],[.5,.5],1920,1080),192)
    def test_model_retains_offscreen_predictions(self):
        x=np.tile(np.linspace(0,1,20)[:,None],(1,8))
        m=GazeModel().fit(x,np.column_stack([x[:,0],x[:,0]]))
        self.assertGreater(m.predict(np.ones(8)*3)[0],1)
    def test_missing_iris_rejected(self):
        self.assertEqual(extract_features([]),(None,'sin_iris'))
    def test_closed_eye_rejected(self):
        points=[SimpleNamespace(x=0.,y=0.) for _ in range(478)]
        points[33].x=.1;points[133].x=.2
        self.assertEqual(extract_features(points)[1],'parpadeo_ojo_cerrado')

if __name__=='__main__':
    unittest.main()
