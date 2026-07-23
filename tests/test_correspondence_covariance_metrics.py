import unittest
import torch
from symm_template_reg.evaluation.correspondence_diagnostics import covariance_geometry


class CovarianceGeometryTest(unittest.TestCase):
    def test_collapsed_cloud_has_zero_rank(self):
        result=covariance_geometry(torch.zeros((12,3)))
        self.assertEqual(int(result["rank"]),0)
        self.assertEqual(float(result["covariance_eigenvalues"].max()),0.)

