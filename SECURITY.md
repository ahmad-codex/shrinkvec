# Security

shrinkvec connects to your database with the connection string you give it. It reads the table you
name and writes only to its own `shrinkvec` schema, which it drops afterwards unless you pass
`--keep`. Reports never contain the connection string.

If you find a way it could change or leak data it should not, please report it privately through
[GitHub security advisories](https://github.com/ahmad-codex/shrinkvec/security/advisories/new), not
as a public issue. You will get a reply within a few days.
