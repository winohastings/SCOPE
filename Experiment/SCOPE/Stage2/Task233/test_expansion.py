import unittest

from Experiment.SCOPE.Stage2.Task233.expand_anchors import route, select_full_train, stats


class ExpansionTests(unittest.TestCase):
    def test_round_robin_and_excluded_instruction_variants(self):
        records = [{"episode_id": str(index), "scene_id": f"mp3d/{scene}/{scene}.glb", "trajectory_id": trajectory}
                   for index, (scene, trajectory) in enumerate([(scene, trajectory) for scene in ("a", "b", "c")
                                                               for trajectory in range(4) for repeat in range(2)])]
        chosen = select_full_train(records, {("a", "0")})
        self.assertEqual(chosen, select_full_train(records, {("a", "0")}))
        self.assertEqual(len(chosen), 11)
        self.assertEqual(len({route(record) for record in chosen}), 11)
        self.assertNotIn(("a", "0"), {route(record) for record in chosen})
        self.assertEqual(len({route(record)[0] for record in chosen[:3]}), 3)
        self.assertEqual(stats(chosen)["scenes"], 3)

    def test_trajectory_identity_is_scene_scoped(self):
        records = [{"scene_id": scene, "trajectory_id": 1} for scene in ("a", "b")]
        self.assertEqual(len(select_full_train(records, set())), 2)


if __name__ == "__main__":
    unittest.main()
