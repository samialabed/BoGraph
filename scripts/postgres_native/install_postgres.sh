sudo su - postgres

psql -U postgres postgres -f ./benchbase_prepare.sql

alias postgres=psql

EXPORT POSTGRES_USER="admin"
EXPORT POSTGRES_PASSWORD="password"
EXPORT POSTGRES_DB="benchbase"
