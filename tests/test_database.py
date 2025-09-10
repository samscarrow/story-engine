
import unittest
import os
import tempfile
import shutil
from story_engine.core.storage.database import SQLiteConnection

class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Set up a temporary, per-test database (xdist-safe)."""
        # Create a unique temp directory per test to avoid xdist collisions
        self.tempdir = tempfile.mkdtemp(prefix="story-db-")
        self.db_name = os.path.join(self.tempdir, "test.db")
        self.db = SQLiteConnection(db_name=self.db_name)
        self.db.connect()

    def tearDown(self):
        """Clean up the temporary database."""
        self.db.disconnect()
        try:
            if os.path.exists(self.db_name):
                os.remove(self.db_name)
        finally:
            # Remove the temp directory tree
            shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_store_and_get_output(self):
        """Test storing and retrieving a workflow output."""
        workflow_name = "test_workflow"
        output_data = {"key": "value", "number": 123}

        # Store the output
        self.db.store_output(workflow_name, output_data)

        # Retrieve the outputs
        retrieved_outputs = self.db.get_outputs(workflow_name)

        # Check that the retrieved output matches the stored data
        self.assertEqual(len(retrieved_outputs), 1)
        self.assertEqual(retrieved_outputs[0], output_data)

    def test_get_outputs_for_nonexistent_workflow(self):
        """Test retrieving outputs for a workflow that doesn't exist."""
        retrieved_outputs = self.db.get_outputs("nonexistent_workflow")
        self.assertEqual(len(retrieved_outputs), 0)

    def test_multiple_outputs(self):
        """Test storing and retrieving multiple outputs for the same workflow."""
        workflow_name = "test_workflow_multiple"
        output_data1 = {"key": "value1"}
        output_data2 = {"key": "value2"}

        self.db.store_output(workflow_name, output_data1)
        self.db.store_output(workflow_name, output_data2)

        retrieved_outputs = self.db.get_outputs(workflow_name)
        self.assertEqual(len(retrieved_outputs), 2)
        self.assertIn(output_data1, retrieved_outputs)
        self.assertIn(output_data2, retrieved_outputs)

if __name__ == '__main__':
    unittest.main()

