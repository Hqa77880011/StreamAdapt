import unittest

from streamadapt.cli import summarize_audit


class AuditTests(unittest.TestCase):
    def test_proxy_improvement_can_still_be_task_harm(self):
        audit = dict(frame=10, difference=[-1., -2.], task_difference=[0.2, 0.4],
                     accuracy_difference=[-1, 0])
        result = summarize_audit(audit)
        self.assertFalse(result["proxy_harmful"])
        self.assertTrue(result["harmful"])
        self.assertTrue(result["accuracy_harmful"])
        self.assertAlmostEqual(result["mean_task_ce_change"], 0.3)

    def test_proxy_harm_does_not_imply_task_harm(self):
        audit = dict(frame=10, difference=[1.], task_difference=[-0.3],
                     accuracy_difference=[1])
        result = summarize_audit(audit)
        self.assertTrue(result["proxy_harmful"])
        self.assertFalse(result["harmful"])
        self.assertFalse(result["accuracy_harmful"])


if __name__ == "__main__":
    unittest.main()
