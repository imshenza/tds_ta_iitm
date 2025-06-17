Data Transformation with dbt
You’ll learn how to transform data using dbt (data build tool), covering:
dbt Fundamentals
: Understand what dbt is and how it brings software engineering practices to data transformation
Project Setup
: Learn how to initialize a dbt project, configure your warehouse connection, and structure your models
Models and Materialization
: Create your first dbt models and understand different materialization strategies (view, table, incremental)
Testing and Documentation
: Implement data quality tests and auto-generate documentation for your data models
Jinja Templating
: Use Jinja for dynamic SQL generation, making your transformations more maintainable and reusable
References and Dependencies
: Learn how to reference other models and manage model dependencies
Sources and Seeds
: Configure source data connections and manage static reference data
Macros and Packages
: Create reusable macros and leverage community packages to extend functionality
Incremental Models
: Optimize performance by only processing new or changed data
Deployment and Orchestration
: Set up dbt Cloud or integrate with Airflow for production deployment
Here’s a minimal dbt model example, 
models/staging/stg_customers.sql
:
with
 source 
as
 
(

    
select
 
*
 
from
 {{ source
(
'raw'
,
 
'customers'
)
 }}

)
,


renamed 
as
 
(

    
select

        id 
as
 customer_id
,

        first_name
,

        last_name
,

        email
,

        created_at
    
from
 source

)



select
 
*
 
from
 renamed
Copy to clipboard
Error
Copied
Tools and Resources:
dbt Core
 - The open-source transformation tool
dbt Cloud
 - Hosted platform for running dbt
dbt Packages
 - Reusable modules from the community
dbt Documentation
 - Comprehensive guides and references
Jaffle Shop
 - Example dbt project for learning
dbt Slack Community
 - Active community for support and discussions
Watch this dbt Fundamentals Course (90 min):














Previous




Parsing JSON












Next










Transforming Images





