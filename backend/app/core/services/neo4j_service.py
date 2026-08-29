import os
from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any
from datetime import datetime

class Neo4jService:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j"""
        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                self.verify_connection()
                print("✅ Connected to Neo4j (Client Backend)")
            except Exception as e:
                # Log usage of deprecated driver or connection failure
                print(f"❌ Failed to connect to Neo4j: {e}")
                self.driver = None

    def is_connected(self):
        return self.driver is not None

    def verify_connection(self):
        """Verify connectivity"""
        if self.driver:
            self.driver.verify_connectivity()

    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def _execute_write(self, query, **kwargs):
        if not self.driver:
            self.connect()
        
        if not self.driver:
            print("⚠️ Skipping Neo4j write: Driver not connected")
            return None

        try:
            with self.driver.session() as session:
                return session.execute_write(lambda tx: tx.run(query, **kwargs).data())
        except Exception as e:
            print(f"❌ Neo4j Write Error: {e}")
            return None

    def _execute_read(self, query, **kwargs):
        if not self.driver:
            self.connect()
            
        if not self.driver:
            print("⚠️ Skipping Neo4j read: Driver not connected")
            return []

        try:
            with self.driver.session() as session:
                return session.execute_read(lambda tx: tx.run(query, **kwargs).data())
        except Exception as e:
            print(f"❌ Neo4j Read Error: {e}")
            return []

    # =========================================================================
    # TIME TREE LOGIC
    # =========================================================================
    def create_time_tree_node(self, timestamp: datetime):
        """
        Create Year -> Month -> Day hierarchy for the given timestamp.
        Returns the Day node to link events to.
        """
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        date_str = timestamp.strftime("%Y-%m-%d")

        query = """
        MERGE (y:Year {year: $year})
        MERGE (m:Month {month: $month, year: $year})
        MERGE (y)-[:HAS_MONTH]->(m)
        MERGE (d:Day {day: $day, month: $month, year: $year, date: $date_str})
        MERGE (m)-[:HAS_DAY]->(d)
        RETURN d
        """
        self._execute_write(query, year=year, month=month, day=day, date_str=date_str)

    # =========================================================================
    # CHAT LOGIC
    # =========================================================================
    def create_session(self, session_id: str, user_id: str, title: str):
        query = """
        MERGE (u:User {userId: $user_id})
        MERGE (s:Session {id: $session_id})
        ON CREATE SET s.title = $title, s.createdAt = datetime()
        MERGE (u)-[:HAS_SESSION]->(s)
        """
        self._execute_write(query, session_id=session_id, user_id=user_id, title=title)

    def add_message(self, session_id: str, role: str, content: str, message_id: str = None):
        """Atomic addition of message to session and time tree"""
        import uuid
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        if not message_id:
            message_id = str(uuid.uuid4())
        
        # Ensure time tree nodes exist first
        self.create_time_tree_node(now)

        query = """
        MATCH (s:Session {id: $session_id})
        MATCH (d:Day {date: $date_str})
        
        MERGE (m:Message {id: $message_id})
        ON CREATE SET 
            m.content = $content,
            m.role = $role,
            m.timestamp = datetime($iso_timestamp)
        
        MERGE (s)-[:CONTAINS]->(m)
        MERGE (m)-[:OCCURRED_ON]->(d)
        
        // Link List Logic
        // TODO: This simple logic might fail if we merge old messages out of order.
        // Assuming sync happens in order or we just link to existing chain.
        // For MVP duplicate prevention, the MERGE is enough to stop double-nodes.
        // The chain linking might be redundant if node exists, but MERGE handles it ok.
        WITH s, m
        OPTIONAL MATCH (s)-[r:LAST_MESSAGE]->(last)
        WHERE last <> m
        DELETE r
        MERGE (s)-[:LAST_MESSAGE]->(m)
        WITH m, last
        WHERE last IS NOT NULL AND last <> m
        MERGE (last)-[:NEXT]->(m)
        """
        self._execute_write(query, session_id=session_id, date_str=date_str, content=content, role=role, iso_timestamp=now.isoformat(), message_id=message_id)

    # =========================================================================
    # ENTITY & TOOL TRACKING
    # =========================================================================
    def create_entity_mention(self, message_id: str, entity_type: str, entity_id: str, context: str = ""):
        """Link message to mentioned entity (Asset, Person, Concept)"""
        query = f"""
        MATCH (m:Message {{id: $message_id}})
        MERGE (e:{entity_type} {{id: $entity_id}})
        MERGE (m)-[:MENTIONS {{context: $context, timestamp: datetime()}}]->(e)
        """
        self._execute_write(
            query,
            message_id=message_id,
            entity_id=entity_id,
            context=context
        )

    def create_tool_usage(self, message_id: str, tool_name: str, params: Dict, result: Dict):
        """Record tool usage in message"""
        import json
        query = """
        MATCH (m:Message {id: $message_id})
        MERGE (t:Tool {name: $tool_name})
        MERGE (m)-[:USED_TOOL {
            params: $params,
            result: $result,
            timestamp: datetime()
        }]->(t)
        """
        self._execute_write(
            query,
            message_id=message_id,
            tool_name=tool_name,
            params=json.dumps(params),
            result=json.dumps(result)
        )

    def create_session_summary(self, session_id: str, summary: str, token_count: int):
        """Create summary node for session"""
        query = """
        MATCH (s:Session {id: $session_id})
        MERGE (sum:Summary {sessionId: $session_id})
        ON CREATE SET sum.createdAt = datetime()
        SET sum.content = $summary,
            sum.tokenCount = $token_count,
            sum.updatedAt = datetime()
        MERGE (s)-[:HAS_SUMMARY]->(sum)
        """
        self._execute_write(
            query,
            session_id=session_id,
            summary=summary,
            token_count=token_count
        )

    def retrieve_graph_context(self, user_id: str, query_entities: List[str], limit: int = 10) -> List[Dict]:
        """Retrieve relevant context from graph based on entities"""
        query = """
        MATCH (u:User {userId: $user_id})-[:HAS_SESSION]->(s:Session)
        -[:CONTAINS]->(m:Message)-[:MENTIONS]->(e)
        WHERE e.id IN $query_entities
        RETURN s.id as session_id, s.title, m.content, m.timestamp, e.id as entity
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        result = self._execute_read(
            query,
            user_id=user_id,
            query_entities=query_entities,
            limit=limit
        )
        return result if result else []

    def get_old_sessions(self, older_than_days: int = 7) -> List[Dict]:
        """Retrieve sessions older than X days that are not archived"""
        query = """
        MATCH (s:Session)
        WHERE s.createdAt < datetime() - duration({days: $days})
        AND (s.archived IS NULL OR s.archived = false)
        RETURN s.id as id, s.title as title, s.createdAt as createdAt
        """
        return self._execute_read(query, days=older_than_days)

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Retrieve all messages for a session in order"""
        query = """
        MATCH (s:Session {id: $session_id})-[:CONTAINS]->(m:Message)
        RETURN m.role as role, m.content as content, m.timestamp as timestamp
        ORDER BY m.timestamp ASC
        """
        return self._execute_read(query, session_id=session_id)

    def archive_session(self, session_id: str, summary: str):
        """Mark session as archived and store summary"""
        query = """
        MATCH (s:Session {id: $session_id})
        SET s.archived = true,
            s.summary = $summary,
            s.archivedAt = datetime()
        """
        self._execute_write(query, session_id=session_id, summary=summary)
        # Also create a summary node for structured graph access
        self.create_session_summary(session_id, summary, len(summary.split()))

# Singleton
_neo4j_service = None

def get_neo4j_service() -> Neo4jService:
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service
