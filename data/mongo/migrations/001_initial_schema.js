db.createCollection("fundus_images", {
    validator: {
       $jsonSchema: {
          bsonType: "object",
          required: ["filename", "diagnosis", "split"],
          properties: {
             filename: { bsonType: "string" },
             diagnosis: { enum: ["normal", "glaucoma"] },
             split: { enum: ["train", "val", "test"] },
             processed: { bsonType: "bool" },
             processed_at: { bsonType: "date" }
          }
       }
    }
 })